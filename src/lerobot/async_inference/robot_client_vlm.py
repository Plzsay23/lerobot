import os
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from queue import Queue
from typing import Any
import numpy as np

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import (
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action, FPSTracker, Observation, RawObservation, RemotePolicyConfig,
    TimedAction, TimedObservation, get_logger, map_robot_keys_to_lerobot_features
)
from .manager_agent import ManagerAgent


def return_to_home(robot, logger=None):
    import time
    def raw_to_deg(raw_val):
        return (raw_val - 2048) * (360.0 / 4096.0)
    
    FINAL_RAW = [2128, 872, 3184, 2708, 1870, 1540]
    FINAL_DEG = [raw_to_deg(v) for v in FINAL_RAW]
    
    try:
        obs = robot.get_observation()
        JOINT_KEYS = list(robot.action_features)
        
        current_pos = []
        for key in JOINT_KEYS:
            if key in obs:
                val = obs[key]
                if hasattr(val, "item"): val = val.item()
                current_pos.append(float(val))
            else:
                return

        steps = 30
        sleep_time = 0.03
        working_pos = list(current_pos)
        sequence_indices = [1, 2, 3] 
        
        for step_idx, joint_idx in enumerate(sequence_indices):
            step_target_pos = list(working_pos)
            step_target_pos[joint_idx] = 0.0  
            
            for i in range(1, steps + 1):
                interpolated_action = []
                for curr, target in zip(working_pos, step_target_pos):
                    interpolated_action.append(curr + (target - curr) * (i / steps))
                
                action_dict = {key: val for key, val in zip(JOINT_KEYS, interpolated_action)}
                robot.send_action(action_dict)
                time.sleep(sleep_time)
                
            working_pos = list(step_target_pos)
            
        for i in range(1, steps + 1):
            interpolated_action = []
            for curr, target in zip(working_pos, FINAL_DEG):
                interpolated_action.append(curr + (target - curr) * (i / steps))
            
            action_dict = {key: val for key, val in zip(JOINT_KEYS, interpolated_action)}
            robot.send_action(action_dict)
            time.sleep(sleep_time)
            
    except Exception:
        pass
        
        
class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig, robot=None):
        self.config = config
        if robot is not None:
            self.robot = robot
        else:
            self.robot = make_robot_from_config(config.robot)
            self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = config.chunk_size_threshold
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock() 
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2) 
        self.current_instruction = getattr(config, "task", "")
        self.fps_tracker = FPSTracker(target_fps=getattr(config, "fps", 30))
        self.must_go = threading.Event()
        self.must_go.set() 
        
    def set_instruction(self, text: str):
        self.current_instruction = text

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError:
            return False

    def stop(self):
        self.shutdown_event.set()
        self.robot.disconnect()
        self.channel.close()

    def send_observation(self, obs: TimedObservation) -> bool:
        if not self.running:
            raise RuntimeError("Client not running.")

        observation_bytes = pickle.dumps(obs)
        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError:
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        return queue_size, timestamps

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction], aggregate_fn=None):
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            if new_action.get_timestep() <= latest_action:
                continue
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        self.start_barrier.wait()

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  

                timed_actions = pickle.loads(actions_chunk.data) 
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()  

            except grpc.RpcError:
                pass

    def actions_available(self):
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        return _performed_action

    def _ready_to_send_observation(self):
        with self.action_queue_lock:
            return self.action_queue.qsize() / max(1, self.action_chunk_size) <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            raw_observation: RawObservation = self.robot.get_observation()
            if self.current_instruction:
                raw_observation["task"] = self.current_instruction
            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(), 
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()

            _ = self.send_observation(observation)

            if observation.must_go:
                self.must_go.clear()

            return raw_observation

        except Exception:
            pass

    def control_loop(self, task: str, max_duration: int = 30, verbose: bool = False) -> tuple[Observation, Action]:
        self.start_barrier.wait()
        _performed_action = None
        _captured_observation = None
        start_time = time.time() 
    
        while self.running:
            if max_duration > 0 and (time.time() - start_time) > max_duration:
                break
            
            control_loop_start = time.perf_counter()
            
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
    
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
    
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    hf_user = os.environ.get("HF_USER", "").strip()
    available_models = []
    
    model_keys = sorted([k for k in os.environ.keys() if k.startswith("MODEL_")])
    for key in model_keys:
        model_name = os.environ.get(key, "").strip()
        if model_name:
            full_path = f"{hf_user}/{model_name}" if hf_user else model_name
            available_models.append(full_path)
            
    if not available_models:
        available_models = ["dummy/model1", "dummy/model2"]

    num_classes = len(available_models) + 1 
    vlm_agent = ManagerAgent(num_classes=num_classes)
    
    shared_robot = make_robot_from_config(cfg.robot)
    shared_robot.connect()
    
    test_obs = shared_robot.get_observation()
    camera_keys = [k for k in test_obs.keys() if "image" in k]
    main_camera_key = camera_keys[0] if camera_keys else None
    
    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue

            obs = shared_robot.get_observation()
            if main_camera_key and main_camera_key in obs:
                current_image_tensor = obs[main_camera_key]
            else:
                current_image_tensor = torch.zeros((3, 480, 640))

            target_idx = vlm_agent.predict_action_index(current_image_tensor, user_input)

            if target_idx == len(available_models):
                continue
            
            selected_model_path = available_models[target_idx]
            cfg.pretrained_name_or_path = selected_model_path

            client = RobotClient(cfg, robot=shared_robot)
            client.set_instruction(user_input)

            if client.start():
                action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
                action_receiver_thread.start()
    
                try:
                    client.control_loop(task=user_input, max_duration=30)
                    return_to_home(shared_robot, client.logger)
    
                except KeyboardInterrupt:
                    return_to_home(shared_robot, client.logger)
    
                finally:
                    client.shutdown_event.set()
                    client.channel.close()
                    if action_receiver_thread.is_alive():
                        action_receiver_thread.join(timeout=1.0)
                        
        except Exception:
            pass
            
    if 'shared_robot' in locals():
        shared_robot.disconnect()

if __name__ == "__main__":
    async_client()