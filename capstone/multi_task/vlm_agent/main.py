# main.py
# 전체 프로그램을 구동하고 구성 요소(환경, 매니저, 모델)를 조립하여 루프를 돌리는 메인 파일

import time
from environment import SimulationEnv
from vla_models import PickPlaceVLA, CleaningVLA
from manager_agent import ManagerAgent

def main():
    # 1. 환경 생성
    env = SimulationEnv()

    # 2. 사용할 VLA 모델들(Experts) 등록
    # 실제 Orin NX에서는 여기서 모델을 VRAM에 로드하거나 경로를 지정함
    vla_registry = {
        'pick_place': PickPlaceVLA(name="Arm_A (Gripper)", specialty="이동"),
        'cleaning': CleaningVLA(name="Arm_B (Sponge)", specialty="청소")
    }

    # 3. 매니저(VLM Agent) 생성
    manager = ManagerAgent(vla_registry)
    
    # 사용자 명령
    user_goal = "테이블을 깨끗이 치우고 닦아줘"
    print("===  Multi-VLA Agent System 시작 ===")

    # 4. 메인 루프 (Observe -> Think -> Act)
    step = 0
    while True:
        step += 1
        print(f"\n--- Step {step} ---")
        time.sleep(1) # 생각하는 척

        # 현재 상태 가져오기
        current_state = env.get_state()

        # [Manager] 판단
        model, target, reason = manager.observe_and_think(current_state, user_goal)

        # 종료 조건
        if model is None:
            print(f" [System] 종료: {reason}")
            break

        print(f" [System] '{model.name}' 선택됨 ({reason})")

        # [Action] 실행 및 상태 업데이트
        new_state = model.execute(current_state, target)
        env.update_state(new_state)

    print("\n===  최종 결과 ===")
    print(env.get_state())

if __name__ == "__main__":
    main()