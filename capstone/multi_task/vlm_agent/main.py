# main.py
import time
import cv2

# 로컬 모듈 임포트 (같은 폴더 내의 vla_models.py, manager_agent.py)
from vla_models import PickPlaceVLA, CleaningVLA
from manager_agent import ManagerAgent

# lerobot 파이프라인 임포트
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

def main():
    print("=== Multi-VLA Agent System 시작 ===")
    
    # 1. Config 객체를 생성하여 카메라 초기화 (0번 카메라 또는 '/dev/video0' 사용)
    print("📷 [System] 카메라를 초기화합니다...")
    camera_config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    camera = OpenCVCamera(camera_config)
    camera.connect()
    
    # 2. VLA 모델 및 매니저(VLM) 로드
    vla_registry = {
        'pick_place': PickPlaceVLA(name="Arm_A (Gripper)", specialty="이동"),
        'cleaning': CleaningVLA(name="Arm_B (Sponge)", specialty="청소")
    }
    manager = ManagerAgent(vla_registry)
    
    print("\n✅ 모든 시스템 준비 완료. 인스트럭션을 대기합니다.")

    # 3. 메인 인터랙티브 루프
    try:
        while True:
            print("\n" + "="*60)
            user_goal = input("⌨️ 사용자 명령을 입력하세요 (종료 'q'): ")
            
            if user_goal.lower() == 'q':
                break
            if not user_goal.strip():
                continue

            # [카메라 캡처] lerobot 파이프라인 방식으로 현재 프레임 가져오기
            print("📷 [카메라 입력] 현재 상황을 촬영 중...")
            image_rgb = camera.read()  # 반환값: RGB 형태의 numpy array
            
            # [VLM 판단] 카메라 이미지와 텍스트를 VLM에 전달
            model, target, reason = manager.observe_and_think(image_rgb, user_goal)

            if model is None:
                print(f"❌ [에러/거절] {reason}")
                continue

            print(f"\n💡 [VLM 최종 결정] '{model.name}' 실행!")
            print(f"   - 목표물: {target}")
            print(f"   - 판단 근거: {reason}")
            time.sleep(0.5)

            # [Action 실행 (문자열 모사)]
            print(f"\n🚀 [로봇 제어] {model.name} 실행 중...")
            time.sleep(2) # 실제 로봇 제어 대기시간
            
            # 환경(env) 객체 없이 모델 내부 로직만 출력하도록 단순화
            dummy_state = {
                'table': ['사과', '물병'], 
                'bin': [], 
                'surface_status': 'dirty'
            }
            model.execute(dummy_state, target)
            
            print(f"✅ [작업 완료] 다음 명령을 대기합니다.")
            
    finally:
        # 안전한 종료 처리
        print("🔌 시스템을 종료하고 카메라를 해제합니다.")
        if camera.is_connected:
            camera.disconnect()

if __name__ == "__main__":
    main()