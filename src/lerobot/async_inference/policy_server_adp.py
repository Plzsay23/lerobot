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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)

# 멀티 어댑터 추가
from .multi_adapter_manager import MultiAdapterManager
import os


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS 및 큐 설정
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self.last_processed_obs = None

        # 모델 및 프로세서 속성
        self.device = config.device if hasattr(config, "device") else "cuda"
        self.policy_type = "xvla" # XVLA 정책 고정
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        # [수정] 멀티 어댑터 매니저 초기화 (이때 어댑터들이 VRAM에 상주됨)
        self.adapter_manager = MultiAdapterManager(self.logger)

        # [수정] 서버 시작 시 베이스 모델을 VRAM에 즉시 상주
        self._initialize_base_model()

    def _initialize_base_model(self):
        """서버 시작 시 베이스 모델을 bfloat16 정밀도로 VRAM에 로드하여 상주시킵니다."""
        base_model_path = os.getenv("BASE_MODEL", "lerobot/xvla-base")
        self.logger.info(f"🚀 [INIT] 베이스 모델 VRAM 상주 시작 (bfloat16): {base_model_path}")
        
        try:
            # 정책 클래스 확보 및 모델 로드
            policy_class = get_policy_class(self.policy_type)
            self.policy = policy_class.from_pretrained(
                base_model_path, 
                dtype="bfloat16" # 하프 프리시전 강제
            )
            self.policy.to(self.device)
            self.policy.eval() # 추론 모드 설정
            
            # 기본 전/후처리기 생성 (장치 최적화)
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=base_model_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
            
            # 매니저에 상주된 정책 객체 전달
            self.adapter_manager.set_policy(self.policy)
            self.policy.config.num_denoising_steps = 3 
            self.logger.info("✅ Denoising steps set to 3 for real-time performance.")
            self.logger.info("✅ 베이스 모델 및 기본 프로세서 VRAM 상주 완료.")
        except Exception as e:
            self.logger.error(f"❌ 베이스 모델 상주 실패: {e}")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        """클라이언트의 새 명령(어댑터 번호 + 텍스트)을 처리합니다."""
        if not self.running:
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)
        instr = str(policy_specs.pretrained_name_or_path).strip()
        
        # 1. 입력 문자열 해석 ('1 pick up' -> ID: 1, Text: 'pick up')
        target_adapter_idx = None
        instruction_text = instr
        if " " in instr:
            parts = instr.split(" ", 1)
            if parts[0].isdigit():
                target_adapter_idx = int(parts[0])
                instruction_text = parts[1]
        elif instr.isdigit():
            target_adapter_idx = int(instr)
            instruction_text = "default task"

        # 2. 메타데이터 업데이트 (필요 시)
        self.lerobot_features = policy_specs.lerobot_features or self.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk or self.actions_per_chunk

        # 3. [병목 제거] 명령이 왔을 때만 어댑터를 갈아끼웁니다.
        if target_adapter_idx is not None:
            start_switch = time.perf_counter()
            self.adapter_manager.switch(target_adapter_idx)
            switch_time = time.perf_counter() - start_switch
            self.logger.info(f"🔄 어댑터 {target_adapter_idx}번으로 교체 완료 ({switch_time:.4f}s)")

        # 4. 텍스트 명령어 저장
        self.current_instruction = instruction_text
        self.logger.info(f"📡 새 태스크 설정: '{instruction_text}'")
        
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """추론 속도 최적화를 위해 denoising steps를 확인합니다."""
        # XVLA의 경우 num_denoising_steps가 성능의 predi핵심입니다.
        # 필요하다면 여기서 강제로 단계를 낮추어 테스트하십시오.
        # self.policy.config.num_denoising_steps = 3 
        
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """순수하게 추론 연산만 수행합니다. (어댑터 교체 로직 없음)"""
        
        # 1. 관측치 데이터 준비
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        # 2. 전처리 및 추론 (병목 감시 로그 포함)
        start_inference = time.perf_counter()
        
        # [디버깅] 현재 설정된 Denoising Steps 확인
        current_steps = self.policy.config.num_denoising_steps
        
        with torch.inference_mode(): #
            # 전처리
            observation = self.preprocessor(observation)
            
            # 모델 추론 (어댑터는 이미 상주된 상태로 연산만 수행)
            action_tensor = self._get_action_chunk(observation)
            
        inference_time = time.perf_counter() - start_inference
        
        # [성능 모니터링] Steps와 시간을 함께 출력
        self.logger.info(
            f"🚀 [Steps: {current_steps}] Inference took {inference_time:.4f}s | "
            f"Shape: {action_tensor.shape}"
        )

        # 3. 후처리 및 TimedAction 변환
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
        with torch.inference_mode():
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        action_tensor = action_tensor.detach().cpu()

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        
        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
