import time
import cv2
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 카메라 클래스 및 설정 임포트
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

FPS = 30

def main():
    # 1. 기존 설정
    robot_config = LeKiwiClientConfig(remote_ip="192.168.0.23", id="lekiwi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/leader", id="leader")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # 2. PC 로컬 카메라 설정 추가
    # index_or_path: 0번이 안되면 1, 2 등으로 변경해 보세요.
    pc_cam_config = OpenCVCameraConfig(
        index_or_path=4, 
        width=960, 
        height=540, 
        fps=FPS,
        warmup_s=2  # 카메라 안정화를 위해 2초 예열
    )
    
    # 객체 생성
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)
    pc_camera = OpenCVCamera(pc_cam_config)

    # 3. 연결 (이 순서가 중요합니다)
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
        # 카메라 연결 실패 시 프로그램을 종료하거나 예외 처리를 할 수 있습니다.

    # 4. 시각화 초기화
    init_rerun(session_name="lekiwi_teleop_with_pc_cam")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected or not pc_camera.is_connected:
        raise ValueError("Robot, Teleop, or PC Camera is not connected!")

    print("Starting teleop loop...")
    try:
        while True:
            t0 = time.perf_counter()

            # 르키위 로봇 관측 데이터 (카메라 2대 포함)
            observation = robot.get_observation()

            # PC 로컬 카메라 데이터 읽기 및 병합
            # async_read는 내부적으로 스레드에서 최신 프레임을 가져옵니다.
            pc_frame = pc_camera.async_read()
            observation["pc_camera"] = pc_frame

            # 제어 로직
            arm_action = leader_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            
            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

            # 로봇에 액션 전송
            _ = robot.send_action(action)

            # Rerun 시각화 (observation에 추가된 pc_camera도 함께 표시됩니다)
            log_rerun_data(observation=observation, action=action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # 자원 해제
        pc_camera.disconnect()
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()

if __name__ == "__main__":
    main()