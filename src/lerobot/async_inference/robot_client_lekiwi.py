import logging
import pickle
import threading
import time
from queue import Queue, Empty
import numpy as np
import torch
import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    FPSTracker,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient  # 수정했던 클라이언트

class LeKiwiAsyncClient(RobotClient):
    """
    LeKiwi 로봇을 위한 범용 Async 클라이언트.
    젯슨(Jetson)에서 실행되며, 모델 서버(PC)와 gRPC로 통신합니다.
    """
    def __init__(self, config: RobotClientConfig):
        # 기존 RobotClient의 __init__ 로직을 LeKiwi에 맞게 커스터마이징
        self.config = config
        
        # 1. LeKiwi 하드웨어 연결 (우리가 수정했던 3개 카메라 버전)
        self.robot = LeKiwiClient(config.robot)
        self.robot.connect()
        self.logger = get_logger(f"lekiwi_client_{config.robot.id}")

        # 2. 모델 서버(gRPC) 설정
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.server_address = config.server_address
        
        from lerobot.async_inference.helpers import RemotePolicyConfig
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )

        self.channel = grpc.insecure_channel(
            self.server_address, 
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        # 3. 비동기 큐 및 동기화 변수
        self.shutdown_event = threading.Event()
        self.latest_action = -1
        self.latest_action_lock = threading.Lock()
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.start_barrier = threading.Barrier(2)
        self.must_go = threading.Event()
        self.must_go.set()
        
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.action_chunk_size = -1
        self._chunk_size_threshold = config.chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False):
        """
        LeKiwiClient.get_observation()을 호출하면 
        이미 front, wrist, pc 카메라가 포함된 3개 영상 데이터가 수집됩니다.
        """
        try:
            start_time = time.perf_counter()

            # 로봇으로부터 관측값 획득 (3개 카메라 + 상태값)
            raw_observation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            # 큐가 비어있으면 즉시 추론을 요청하도록 설정
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
            
            # gRPC를 통해 서버로 전송
            self.send_observation(observation)

            if observation.must_go:
                self.must_go.clear()

            if verbose:
                self.logger.debug(f"Sent Obs #{observation.get_timestep()} with keys: {raw_observation.keys()}")

            return raw_observation
        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

# 실행을 위한 draccus 래퍼
@draccus.wrap()
def main(cfg: RobotClientConfig):
    client = LeKiwiAsyncClient(cfg)
    if client.start():
        action_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_thread.start()
        try:
            client.control_loop(task=cfg.task, verbose=True)
        finally:
            client.stop()
            action_thread.join()

if __name__ == "__main__":
    main()