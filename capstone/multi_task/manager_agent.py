# manager_agent.py
# 환경의 상태와 사용자의 목표를 비교하여, 어떤 VLA 모델을 호출할지 결정하는 라우터

class ManagerAgent:
    def __init__(self, available_models):
        self.models = available_models

    def observe_and_think(self, env_state, user_goal):
        """
        상황(State)을 보고 적절한 VLA 모델을 선택하여 반환
        """
        print(f"\n [Manager] 상황 판단 중... (목표: {user_goal})")
        print(f"   - 현재 관측: 테이블 위{env_state['table']}, 표면={env_state['surface_status']}")

        # --- 판단 로직 ---
        
        # 1. 테이블 위에 물건이 있으면 -> PickPlaceVLA
        if len(env_state['table']) > 0:
            target = env_state['table'][0]
            selected_model = self.models['pick_place']
            reason = f"테이블 위에 '{target}'이(가) 있음"
            return selected_model, target, reason

        # 2. 물건은 없는데 더러우면 -> CleaningVLA
        elif env_state['surface_status'] == 'dirty':
            selected_model = self.models['cleaning']
            reason = "물건은 없지만 테이블이 더러움"
            return selected_model, "table_surface", reason

        # 3. 할 일 없음
        else:
            return None, None, "모든 작업 완료"