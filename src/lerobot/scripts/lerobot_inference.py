
"""
Inference-only script for running a pretrained policy on a robot.

Example:
```shell
lerobot-inference \
--robot.type=so101_follower \
--robot.port=/dev/so101_follower \
--robot.id=follower \
--robot.cameras='{
      top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 25},
      wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 25},
  }' \
--policy.path=roboseasy/pick_and_place \
--instruction="Pick up the pen and place it in the pencil case" \
--display_data=true
```
"""

import logging
import time
from dataclasses import dataclass

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
    koch_follower,
    omx_follower,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class InferenceConfig:
    robot: RobotConfig
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    instruction: str = ""
    fps: int = 25

    def __post_init__(self):
        # Load pretrained policy configuration if path is provided
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("A policy must be provided via --policy.path or --policy.type")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def main(cfg: InferenceConfig):
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="inference")

    if not cfg.policy.pretrained_path:
        raise ValueError("--policy.path must be provided for inference.")

    # Initialize robot
    robot = make_robot_from_config(cfg.robot)

    # Create robot action and observation processors
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Construct features from robot (no dataset needed)
    robot_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # Load pretrained policy directly (no ds_meta needed)
    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=cfg.policy.pretrained_path,
        config=cfg.policy,
    )
    policy.to(cfg.policy.device)

    # Create preprocessor and postprocessor from pretrained
    # (normalization stats are saved in the processor config, no dataset_stats needed)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
    )

    device = get_safe_torch_device(policy.config.device)

    logging.info(f"Instruction: {cfg.instruction}")

    # Reset policy and processors
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    # Connect to robot
    robot.connect()

    # Initialize keyboard listener for graceful exit
    listener, events = init_keyboard_listener()

    try:
        logging.info("Starting inference loop. Press ESC to stop.")

        while not events.get("stop_recording", False):
            start_loop_t = time.perf_counter()

            if events.get("exit_early", False):
                logging.info("Early exit requested.")
                break

            # Get robot observation
            obs = robot.get_observation()

            # Apply observation processor pipeline (default is IdentityProcessor)
            obs_processed = robot_observation_processor(obs)

            # Build observation frame from processed observations
            observation_frame = build_dataset_frame(robot_features, obs_processed, prefix=OBS_STR)

            # Predict action using the policy
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=getattr(policy.config, "use_amp", False),
                task=cfg.instruction,
                robot_type=cfg.robot.type,
            )

            # DEBUG: Print action values
            logging.info("-" * 80)
            logging.info(f"DEBUG: action_values shape: {action_values.shape}")
            logging.info(f"DEBUG: action_values: {action_values}")
            logging.info("=" * 80)

            # Convert policy action to robot action format
            act_processed_policy = make_robot_action(action_values, robot_features)
            logging.info(f"DEBUG: act_processed_policy: {act_processed_policy}")

            # Apply robot action processor pipeline
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            logging.info(f"DEBUG: robot_action_to_send: {robot_action_to_send}")

            # Send action to robot
            robot.send_action(robot_action_to_send)

            # Log data for visualization if enabled
            if cfg.display_data:
                # action_values는 Tensor이므로 딕셔너리로 변환한 act_processed_policy를 사용
                log_rerun_data(observation=obs_processed, action=act_processed_policy)

            # Maintain target FPS
            dt_s = time.perf_counter() - start_loop_t
            sleep_time = max(0, 1 / cfg.fps - dt_s)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise
    finally:
        # Cleanup
        logging.info("Disconnecting robot...")
        robot.disconnect()

        if listener is not None:
            listener.stop()

        logging.info("Inference completed.")


if __name__ == "__main__":
    main()
