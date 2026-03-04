import logging
import pickle  
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
    services_pb2,  
    services_pb2_grpc,  
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


class PolicyServerVLM(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server_vlm"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self.last_processed_obs = None
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None

        self.cached_policies = {}
        self.cached_preprocessors = {}
        self.cached_postprocessors = {}

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  
        if not self.running:
            return services_pb2.Empty()

        policy_specs = pickle.loads(request.data)  
        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type 
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        model_path = policy_specs.pretrained_name_or_path

        if model_path in self.cached_policies:
            self.policy = self.cached_policies[model_path]
            self.preprocessor = self.cached_preprocessors[model_path]
            self.postprocessor = self.cached_postprocessors[model_path]
            return services_pb2.Empty()

        policy_class = get_policy_class(self.policy_type)
        policy = policy_class.from_pretrained(model_path)
        policy.to(self.device)
        policy.eval()

        device_override = {"device": self.device}
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.cached_policies[model_path] = policy
        self.cached_preprocessors[model_path] = preprocessor
        self.cached_postprocessors[model_path] = postprocessor

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        ) 
        timed_observation = pickle.loads(received_bytes)  
        self._enqueue_observation(timed_observation)
        return services_pb2.Empty()

    def GetActions(self, request, context): 
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            action_chunk = self._predict_action_chunk(obs)
            actions_bytes = pickle.dumps(action_chunk) 
            actions = services_pb2.Actions(data=actions_bytes)

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  

            return actions

        except Empty:  
            return services_pb2.Empty()
        except Exception:
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            return False
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            return False
        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
            self.observation_queue.put(obs)
            return True
        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0) 
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        action_tensor = self._get_action_chunk(observation)
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
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
        self._reset_server()

@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    policy_server = PolicyServerVLM(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()