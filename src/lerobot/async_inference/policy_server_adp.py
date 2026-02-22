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
        base_model_path = os.getenv("BASE_MODEL", "lerobot/xvla-base")
        self.logger.info(f"🚀 [VRAM INIT] {base_model_path}")
        
        try:
            policy_class = get_policy_class(self.policy_type)
            # 1. 모델 로드 시도
            self.policy = policy_class.from_pretrained(base_model_path)
            
            # 2. [강제 전환] 모델 전체를 bfloat16으로 즉시 변경
            self.policy.to(device=self.device, dtype=torch.bfloat16)
            
            # 3. Steps 강제 고정
            self.policy.config.num_denoising_steps = 3
            self.policy.eval()

            # [디버깅 로그 재확인]
            param_dtype = next(self.policy.parameters()).dtype
            vram_usage = torch.cuda.memory_allocated(self.device) / 1024**2
            self.logger.info(f"🔍 [FIXED CHECK] Dtype: {param_dtype} | VRAM: {vram_usage:.2f}MB")
            
            # 전/후처리기 초기화
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=base_model_path,
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
            
            self.adapter_manager.set_policy(self.policy)
            self.logger.info("✅ 베이스 모델 상주 완료.")
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {e}")
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
        """클라이언트 명령 수신 및 로봇 특징에 따른 프로세서 재구성"""
        if not self.running: return services_pb2.Empty()
        policy_specs = pickle.loads(request.data)
        instr = str(policy_specs.pretrained_name_or_path).strip()

        # [핵심 수정] 로봇 특징(관절 정보) 강제 업데이트 및 프로세서 재설정
        if policy_specs.lerobot_features:
            self.lerobot_features = policy_specs.lerobot_features
            device_override = {"device": self.device}
            # 클라이언트 하드웨어 정보에 맞춰 전/후처리기를 다시 생성하여 데이터 규격을 일치시킵니다.
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=os.getenv("BASE_MODEL", "lerobot/xvla-base"),
                preprocessor_overrides={"device_processor": device_override},
                postprocessor_overrides={"device_processor": device_override},
            )
            self.logger.info(f"✅ 로봇 관절 특징 로드 및 프로세서 재설정 완료: {list(self.lerobot_features.keys())}")
        
        target_adapter_idx = None
        if " " in instr:
            parts = instr.split(" ", 1)
            if parts[0].isdigit(): target_adapter_idx = int(parts[0])
        elif instr.isdigit(): target_adapter_idx = int(instr)

        if target_adapter_idx is not None:
            # 1. 어댑터 가중치 스위칭
            self.adapter_manager.switch(target_adapter_idx)
            
            # 2. 어댑터별 stats.json 주입
            env_key = f"ADP{target_adapter_idx}"
            adapter_name = os.getenv(env_key)

            if not adapter_name:
                self.logger.error(f"❌ 환경 변수 {env_key} 설정 누락")
                return services_pb2.Empty()

            stats_path = f"./adapter/{adapter_name}/stats.json" 
            if os.path.exists(stats_path):
                import json
                with open(stats_path, "r") as f:
                    new_stats = json.load(f)

                if self.postprocessor and hasattr(self.postprocessor, 'processors'):
                    for p in self.postprocessor.processors:
                        if hasattr(p, 'stats'):
                            p.stats = {k: torch.tensor(v).to(self.device) for k, v in new_stats.items()}
                            self.logger.info(f"🔥 [SUCCESS] '{adapter_name}' 전용 통계치 주입 완료")
            else:
                self.logger.warning(f"⚠️ {stats_path} 누락으로 기본 통계치를 사용합니다.")

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
        """클라이언트에게 액션 청크를 반환합니다. Dtype 충돌을 정밀 수사합니다."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            # 1. 추론 수행 (내부에서 BFloat16 -> Float32 변환 수행됨)
            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            # 2. [범인 검거] 직렬화 직전 리스트 내 텐서 타입 전수 조사
            if action_chunk and len(action_chunk) > 0:
                # 첫 번째 액션의 실제 데이터 확인
                sample_act = action_chunk[0].action # TimedAction.action 접근
                
                if torch.is_tensor(sample_act):
                    self.logger.info(f"🚨 [DTYPE CHECK] 최종 송신 텐서 타입: {sample_act.dtype}")
                    
                    # 만약 여전히 bfloat16이면 여기서 강제로 float32로 교정 (최후의 보루)
                    if sample_act.dtype == torch.bfloat16:
                        self.logger.warning("⚠️ BFloat16 발견! 피클 직렬화 전 Float32로 강제 변환합니다.")
                        for ta in action_chunk:
                            if torch.is_tensor(ta.action):
                                ta.action = ta.action.to(torch.float32)

            # 3. 직렬화 및 전송
            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)
            serialize_time = time.perf_counter() - start_time

            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            # 지연 시간 조절
            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )

            return actions

        except Empty:
            return services_pb2.Empty()

        except Exception as e:
            # 에러 발생 시 상세 정보 출력
            self.logger.error(f"💥 GetActions 에러 발생: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        """추론 수행 및 물리 수치 검증/보정"""
        try:
            obs_raw = observation_t.get_observation()
            # 관측치를 모델 규격에 맞춰 변환
            observation = raw_observation_to_observation(
                obs_raw, self.lerobot_features, self.policy.config.image_features
            )
            
            with torch.inference_mode():
                # 1. 전처리 수행 (재설정된 preprocessor 사용)
                observation = self.preprocessor(observation)
                
                # 2. 모델 추론 (Autocast 적용)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    adp_out = self.policy.predict_action_chunk(observation)
                
                # [모니터링 1] 모델 원본 출력 스케일 확인
                raw_max = adp_out.abs().max().item()
                self.logger.info(f"🔍 [MONITOR-RAW] Max: {raw_max:.8f}")

                # 3. 후처리 (물리량 복원)
                action_tensor = adp_out.to(dtype=torch.float32)
                processed_actions = []
                for i in range(action_tensor.shape[1]):
                    processed_action = self.postprocessor(action_tensor[:, i, :])
                    processed_actions.append(processed_action)
                
                final_chunk = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu()
                
                # [모니터링 2] 최종 물리 수치 (보정 전)
                self.logger.info(f"📊 [MONITOR-PHYSICAL] J1-J6 (Pre-boost): {[f'{x:.2f}' for x in final_chunk[0].tolist()[:6]]}")
                
                # 하드웨어 보호를 위해 각도를 ±45도로 강제 제한합니다.
                final_chunk = torch.clamp(final_chunk, min=-45.0, max=45.0)
                
                # [모니터링 3] 로봇에 전송될 최종 수치
                final_vals = final_chunk[0].tolist()
                self.logger.info(f"🚨 [FINAL-ACTION] J1-J6: {[f'{x:.2f}' for x in final_vals[:6]]}")

                return self._time_action_chunk(
                    observation_t.get_timestamp(), list(final_chunk), observation_t.get_timestep()
                )

        except Exception as e:
            self.logger.error(f"❌ _predict_action_chunk 추론 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise e

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
