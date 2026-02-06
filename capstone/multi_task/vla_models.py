# vla_models.py
# 실제 물리적인 작업(Action)을 수행하는 모델들

class BaseVLA:
    """모든 VLA 모델의 부모 클래스"""
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty

    def execute(self, env_state, target_obj):
        print(f"   [⚡ {self.name}] 가동 시작... (특기: {self.specialty})")
        return env_state

class PickPlaceVLA(BaseVLA):
    """물건 집기/옮기기 모델"""
    def execute(self, env_state, target_obj):
        super().execute(env_state, target_obj)
        # 시뮬레이션 로직
        if target_obj in env_state['table']:
            print(f"   -> '{target_obj}'을(를) 집어서 정리함으로 이동했습니다.")
            env_state['table'].remove(target_obj)
            env_state['bin'].append(target_obj)
        else:
            print(f"   -> 오류: '{target_obj}'을(를) 찾을 수 없습니다.")
        return env_state

class CleaningVLA(BaseVLA):
    """청소 모델"""
    def execute(self, env_state, target_obj):
        super().execute(env_state, target_obj)
        if env_state['surface_status'] == 'dirty':
            print(f"   -> 테이블 표면을 닦았습니다.")
            env_state['surface_status'] = 'clean'
        else:
            print(f"   -> 이미 깨끗합니다.")
        return env_state