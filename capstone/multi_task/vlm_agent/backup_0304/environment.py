# environment.py

class SimulationEnv:
    def __init__(self):
        # 초기 상태 정의
        self.state = {
            'table': ['사과', '드라이버', '물병'], 
            'bin': [],
            'surface_status': 'dirty'
        }

    def get_state(self):
        """현재 상태 반환"""
        return self.state

    def update_state(self, new_state):
        """상태 업데이트"""
        self.state = new_state