# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any
import numpy as np

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)

def move_joint_smoothly(robot_arm, joint_name, target_pos, steps=100, delay=0.015):
    """특정 조인트를 부드럽게 선형 보간으로 이동시킵니다. (LeRobot 호환)"""
    try:
        # 1. LeRobot API로 현재 딕셔너리 상태 가져오기
        obs = robot_arm.get_observation()
        
        # (도우미 함수) PyTorch Tensor, Numpy Array, 리스트 등 어떤 값이 오든 스칼라(float)로 안전하게 변환
        def _get_float(v):
            if hasattr(v, "item"): return float(v.item())
            if hasattr(v, "__len__") and not isinstance(v, str): return float(v[0])
            return float(v)

        # 2. 현재 움직일 관절의 위치 파악
        if joint_name in obs:
            current_pos = _get_float(obs[joint_name])
        elif f"observation.{joint_name}" in obs:
            current_pos = _get_float(obs[f"observation.{joint_name}"])
        else:
            raise KeyError(f"'{joint_name}' 키를 찾을 수 없습니다. (현재 존재하는 키: {list(obs.keys())})")
            
        # 3. 선형 보간 궤적 생성
        trajectory = np.linspace(current_pos, target_pos, steps)
        print(f"▶ [{joint_name}] 이동: {current_pos:.1f} -> {target_pos}")
        
        # 4. send_action에 넣을 전체 관절 키 리스트 추출 (사용자 원본 코드 로직 응용)
        action_keys = list(robot_arm.action_features.keys()) if isinstance(robot_arm.action_features, dict) else list(robot_arm.action_features)
        
        # 5. 궤적을 따라 이동 수행
        for pos in trajectory:
            action_dict = {}
            for k in action_keys:
                if k == joint_name:
                    # 목표 관절은 궤적(trajectory)의 값을 넣음
                    action_dict[k] = float(pos)
                else:
                    # 목표가 아닌 나머지 관절들은 현재 위치를 그대로 복사해서 넣음
                    obs_k = k if k in obs else f"observation.{k}"
                    action_dict[k] = _get_float(obs.get(obs_k, 0.0))
            
            # 완성된 딕셔너리를 전송!
            robot_arm.send_action(action_dict)
            time.sleep(delay)
            
    except Exception as e:
        print(f"❌ [{joint_name}] 제어 중 오류 발생: {e}")


def return_to_home(robot, logger=None):
    """검증된 딕셔너리 매핑 방식을 사용한 순차적 복귀 함수"""
    import time
    log_func = logger.info if logger else print

    log_func("\n=== 🤖 Return to Home (순차적 복귀) 시작 ===")

    try:
        # 1. 로봇이 현재 가지고 있는 모터 이름 싹 다 가져오기 (원래 코드 방식)
        JOINT_KEYS = list(robot.action_features)
        obs = robot.get_observation()
        
        # 2. 현재 모든 모터의 위치를 딕셔너리로 안전하게 추출
        current_state = {}
        for key in JOINT_KEYS:
            if key in obs:
                val = obs[key]
                if hasattr(val, "item"): val = val.item() # 텐서 처리
                current_state[key] = float(val)
            else:
                log_func(f"❌ 앗! '{key}' 값을 찾을 수 없습니다. 복귀를 취소합니다.")
                return

        log_func(f"📊 현재 위치: [ {', '.join(f'{k}: {v:.1f}' for k, v in current_state.items())} ]")

        # ---------------------------------------------------------
        # 💡 핵심: 특정 모터만 목표값으로 천천히 이동시키는 내부 함수
        # ---------------------------------------------------------
        def move_single_joint(target_joint_name, target_value, steps=30, sleep_time=0.02):
            if target_joint_name not in current_state:
                log_func(f"⚠️ '{target_joint_name}' 모터를 찾을 수 없어 건너뜁니다.")
                return

            log_func(f"▶ [{target_joint_name}] 이동 중: {current_state[target_joint_name]:.1f} -> {target_value}")

            # 목표 상태(Target State) 딕셔너리 만들기 (움직일 관절만 값 변경)
            target_state = current_state.copy()
            target_state[target_joint_name] = target_value

            # 스텝 수만큼 쪼개서 부드럽게 이동
            for i in range(1, steps + 1):
                action_dict = {}
                for key in JOINT_KEYS:
                    start_val = current_state[key]
                    end_val = target_state[key]
                    # 선형 보간 (현재 위치에서 목표 위치로 쪼개서 전송)
                    action_dict[key] = start_val + (end_val - start_val) * (i / steps)

                # 원래 작동하던 그 완벽한 방식으로 전송!
                robot.send_action(action_dict)
                time.sleep(sleep_time)

            # 이동 완료 후, 현재 상태 업데이트
            current_state[target_joint_name] = float(target_value)

        # ---------------------------------------------------------
        # 🧗 3단계 순차적 제어 실행
        # ---------------------------------------------------------
        # step 수와 sleep_time을 조절하여 속도를 제어할 수 있습니다.
        # 예: steps=40, sleep_time=0.02 이면 한 관절당 약 0.8초 소요
        
        # 1. 어깨 먼저 들어올리기
        move_single_joint("shoulder_lift", 2050.0, steps=40)
        
        # 2. 어깨 완료 후 팔꿈치 접기
        move_single_joint("elbow_flex", 2050.0, steps=40)
        
        # 3. 팔꿈치 완료 후 손목 정렬
        move_single_joint("wrist_flex", 2000.0, steps=40)

        log_func("=== ✅ 순차적 홈 복귀 완료. 수고하셨습니다! ===")

    except Exception as e:
        log_func(f"❌ 홈 복귀 중 에러 발생: {e}")
        
class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig, robot =None):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        if robot is not None:
            self.robot = robot
        else:
            self.robot = make_robot_form_config(config.robot)
            self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
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
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # [수정] 동적 명령어 변수 추가
        # 초기값은 실행 옵션(--task)으로 들어온 값을 사용하되, 없으면 빈 문자열
        self.current_instruction = getattr(config, "task", "")
        self.logger.info(f"Initial instruction set to: '{self.current_instruction}'")

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing
        
    def set_instruction(self, text: str):
        self.logger.info(f"Instruction updated: '{text}'")
        self.current_instruction = text

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
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
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            if self.current_instruction:
                raw_observation["task"] = self.current_instruction
            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, max_duration: int = 30, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        self.start_barrier.wait()
        self.logger.info(f"Control loop thread starting (Max Duration: {max_duration}s)")
    
        _performed_action = None
        _captured_observation = None
    
        start_time = time.time()  # 시작 시간 기록
    
        while self.running:
            # [추가된 로직] 지정된 시간 초과 시 루프 탈출
            if max_duration > 0 and (time.time() - start_time) > max_duration:
                self.logger.info(f" 타임아웃({max_duration}초) 도달. 작업을 종료합니다.")
                break
            
            control_loop_start = time.perf_counter()
            
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
    
            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
    
            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action

@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    # 1. 로봇 하드웨어 초기화 (프로그램 켤 때 딱 한 번만 수행)
    print("\n[System] Initializing Robot Hardware...")
    from lerobot.robots import make_robot_from_config
    
    # 로봇 객체를 여기서 미리 생성하고 연결합니다.
    shared_robot = make_robot_from_config(cfg.robot)
    shared_robot.connect()
    
    print("\n" + "="*50)
    print("Interactive Agent Ready!")
    print("==================================================")

    # 2. 메인 루프 (입력 -> 실행 -> 중단 -> 반복)
    while True:
        try:
            print("\n[VLM 대기] 태스크를 입력하세요 (형식: '[번호] [내용]', 예: '1 pick up')")
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input:
                continue

            # 서버가 최초에 모델을 만들 때 필요한 필수 설정들을 그대로 유지
            # cfg.pretrained_name_or_path = user_input
            
            print(f"📡 VLM에 명령 전송 중: '{user_input}'")

            # 클라이언트 생성 시 기존 cfg(policy_type 등 포함)를 그대로 사용
            client = RobotClient(cfg, robot=shared_robot)
            client.set_instruction(user_input)

            if client.start():
                action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
                action_receiver_thread.start()
    
                try:
                    # 제어 루프 진입 (30초 타임아웃 설정)
                    client.control_loop(task=user_input, max_duration=30)
    
                    #  [상황 1 & 2] 루프가 무사히 끝났거나, 타임아웃으로 끝남 (성공/실패 판단 후 복귀)
                    print("\\n 작업(또는 시간)이 종료되었습니다. 초기 위치로 복귀합니다.")
                    return_to_home(shared_robot, client.logger)
    
                except KeyboardInterrupt:
                    #  [상황 3] 사람이 직접 중단함 (긴급 정지)
                    print("\\n 긴급 정지 감지! 로봇을 안전하게 초기 위치로 복귀시킵니다.")
                    return_to_home(shared_robot, client.logger)
    
                finally:
                    # 통신 종료 (모델은 서버 램에 유지)
                    client.shutdown_event.set()
                    client.channel.close()
                    if action_receiver_thread.is_alive():
                        action_receiver_thread.join(timeout=1.0)
                        
        except Exception as e:
            print(f"오류 발생: {e}")
            
    # 프로그램 종료 시 로봇 연결 해제
    if 'shared_robot' in locals():
        shared_robot.disconnect()

if __name__ == "__main__":
    async_client()  # run the client
