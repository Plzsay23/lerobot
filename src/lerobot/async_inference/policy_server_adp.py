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
        """서버 기동 시 베이스 모델을 VRAM에 상주시키고 타입을 점검합니다."""
        base_model_path = os.getenv("BASE_MODEL", "lerobot/xvla-base")
        self.logger.info(f"🚀 [VRAM INIT] 베이스 모델 상주 시작: {base_model_path}")
        
        try:
            from lerobot.policies.xvla.configuration_xvla import XVLAConfig
            import json
            import inspect
            from huggingface_hub import hf_hub_download
            
            # 1. 실제 config.json 다운로드 및 로드
            config_file = hf_hub_download(repo_id=base_model_path, filename="config.json")
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            
            # 2. [핵심] XVLAConfig 클래스에 정의된 필드만 필터링 (가장 안전한 방법)
            # 클래스의 __init__ 인자 목록을 가져와서 딕셔너리와 매칭되는 것만 남깁니다.
            sig = inspect.signature(XVLAConfig.__init__)
            valid_params = [p.name for p in sig.parameters.values() if p.name != 'self']
            
            # 클래스 정의에 있는 키값만 골라내어 새로운 딕셔너리 생성
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
            
            # 3. 필수 설정값 강제 주입
            config = XVLAConfig(**filtered_config)
            config.num_denoising_steps = 3  # 실시간 속도 최적화
            # 만약 차원 에러가 날 경우를 대비해 여기서 6으로 테스트해볼 수도 있습니다.
            # config.max_action_dim = 6 
            
            # 4. 모델 로드 (bfloat16 적용)
            policy_class = get_policy_class(self.policy_type)
            self.policy = policy_class.from_pretrained(
                base_model_path, 
                config=config,
                dtype="bfloat16" # 하프 프리시전 강제
            )
            self.policy.to(self.device)
            self.policy.eval()
            
            # [디버깅] 실제 로드된 모델의 상태 확인
            param_dtype = next(self.policy.parameters()).dtype
            vram_usage = torch.cuda.memory_allocated(self.device) / 1024**2
            self.logger.info(f"🔍 [MODEL CHECK] Dtype: {param_dtype} | VRAM: {vram_usage:.2f}MB")
            
            # 전/후처리기 초기화
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=base_model_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
            
            self.adapter_manager.set_policy(self.policy)
            self.logger.info("✅ 베이스 모델 및 프로세서 상주 완료.")
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
            # 에러 발생 시 상세 트레이스백을 찍어서 원인을 파악합니다.
            import traceback
            self.logger.error(traceback.format_exc())

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
        """클라이언트 명령 수신 시 어댑터를 핫스왑합니다. (추론 루프와 분리)"""
        if not self.running:
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)
        instr = str(policy_specs.pretrained_name_or_path).strip()
        
        # 1. 입력 해석 ('1 pick up')
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

        # 2. 메타데이터 업데이트
        self.lerobot_features = policy_specs.lerobot_features or self.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk or self.actions_per_chunk

        # 3. [최적화] 명령이 올 때만 어댑터 교체 수행
        if target_adapter_idx is not None:
            start_switch = time.perf_counter()
            self.adapter_manager.switch(target_adapter_idx)
            self.logger.info(f"🔄 어댑터 {target_adapter_idx}번 교체 완료 ({time.perf_counter()-start_switch:.4f}s)")

        self.current_instruction = instruction_text
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
        """추론 전과정을 디버깅하고 차원을 슬라이싱합니다."""
        t_start = time.perf_counter()

        # 1. 데이터 준비
        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        t_prepare = time.perf_counter()

        # 2. 전처리 (Dtype 감시)
        with torch.inference_mode():
            observation = self.preprocessor(observation)
        t_preprocess = time.perf_counter()

        # 3. 순수 모델 추론
        with torch.inference_mode():
            # 실제 모델이 뱉는 raw 텐서 (1, 30, 20)
            action_tensor = self.policy.predict_action_chunk(observation)
        t_inference = time.perf_counter()

        # 4. 차원 슬라이싱 (20 -> 6) 및 후처리
        actual_dim = 6 # 기본값 6자유도
        if self.lerobot_features and "action" in self.lerobot_features:
            actual_dim = len(self.lerobot_features["action"])
        
        # [핵심] 로봇 규격에 맞춰 강제 절단
        action_tensor = action_tensor[:, :, :actual_dim]

        processed_actions = []
        with torch.inference_mode():
            for i in range(action_tensor.shape[1]):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)
        
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()
        t_post = time.perf_counter()

        # 5. [성능/데이터 모니터링] 로봇이 안 움직이는 원인 파악용 로그
        act_max = action_tensor.abs().max().item()
        self.logger.info(
            f"⏱️ [Total {1000*(t_post-t_start):.1f}ms] "
            f"Prep: {1000*(t_preprocess-t_prepare):.1f}ms | "
            f"Inf: {1000*(t_inference-t_preprocess):.1f}ms | "
            f"Post: {1000*(t_post-t_inference):.1f}ms"
        )
        self.logger.info(f"📉 [Output] Shape: {action_tensor.shape} | Max Val: {act_max:.4f}")

        if act_max < 1e-4:
            self.logger.error("❌ 액션 값이 0에 가깝습니다. 어댑터 주입이나 Dtype 변환을 확인하세요.")

        return self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )

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
