import time
import cv2
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

FPS = 30

def main():
    robot_config = LeKiwiClientConfig(remote_ip="192.168.0.23", id="lekiwi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/leader", id="leader")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    pc_cam_config = OpenCVCameraConfig(
        index_or_path=4, 
        width=960, 
        height=540, 
        fps=FPS,
        warmup_s=2
    )
    
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)
    pc_camera = OpenCVCamera(pc_cam_config)

    print("Connecting to robot...")
    robot.connect()
    print("Connecting to leader arm...")
    leader_arm.connect()
    print("Connecting to keyboard...")
    keyboard.connect()
    
    print("Connecting to PC local camera...")
    try:
        pc_camera.connect()
    except Exception as e:
        print(f"Failed to connect PC Camera: {e}")

    # 4. 시각화 초기화
    init_rerun(session_name="lekiwi_teleop_with_pc_cam")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected or not pc_camera.is_connected:
        raise ValueError("Robot, Teleop, or PC Camera is not connected!")

    print("Starting teleop loop...")
    try:
        while True:
            t0 = time.perf_counter()

            observation = robot.get_observation()

            pc_frame = pc_camera.async_read()
            observation["pc_camera"] = pc_frame

            arm_action = leader_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            
            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

            _ = robot.send_action(action)

            log_rerun_data(observation=observation, action=action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pc_camera.disconnect()
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()

if __name__ == "__main__":
    main()