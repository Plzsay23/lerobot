import argparse
from lerobot.robots.factory import make_robot_from_config
from lerobot.robots.config import RobotConfig

def release_robot_torque(robot_type, port):
    print(f"🔄 {robot_type} 로봇({port})의 토크 해제를 시작합니다...")
    
    try:
        # 1. 로봇 설정 생성 및 인스턴스화
        # 사용 중인 로봇 타입에 맞게 설정 (예: 'so100', 'koch' 등)
        cfg = RobotConfig(type=robot_type, port=port)
        robot = make_robot_from_config(cfg)
        
        # 2. 하드웨어 연결
        robot.connect()
        
        # 3. 모든 모터 버스의 토크 비활성화 호출
        # SerialMotorsBus.disable_torque() 메서드를 사용하여 전압 차단
        if hasattr(robot, 'motor_bus'):
            robot.motor_bus.disable_torque()
            print("✅ 모든 모터의 토크가 해제되었습니다. 이제 손으로 움직일 수 있습니다.")
        else:
            # 로봇 객체 자체에 토크 해제 명령이 있는 경우
            robot.disable_torque()
            print("✅ 로봇 토크 해제 완료.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        # 4. 안전하게 연결 종료 (이미 위에서 꺼졌지만 확실히 하기 위함)
        if 'robot' in locals() and robot.is_connected:
            robot.disconnect(disable_torque=True)
            print("🔌 연결이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="로봇 타입 (예: so100_follower, koch_leader 등)")
    parser.add_argument("--port", type=str, required=True, help="포트 경로 (예: /dev/ttyACM0)")
    
    args = parser.parse_args()
    release_robot_torque(args.type, args.port)