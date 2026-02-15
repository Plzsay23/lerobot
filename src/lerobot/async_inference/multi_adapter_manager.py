import os
import torch

class MultiAdapterManager:
    def __init__(self, logger):
        self.logger = logger
        self.policy = None
        self.adapter_cache = {}
        
        # 환경 변수로부터 어댑터 경로 동적 생성
        hf_user = os.getenv("HF_USER", "default_user")
        
        # 어댑터 맵 정의 (환경 변수 기반)
        # 예: ADP1="pap_black_xvla_adapter" 로 설정되어 있어야 함
        self.adapter_configs = {
            1: os.getenv("ADP1"),
            2: os.getenv("ADP2"),
            3: os.getenv("ADP3"),
        }
        
        self._load_into_ram()

    def _load_into_ram(self):
        """서버 시작 시 어댑터들을 CPU RAM에 미리 로드합니다."""
        for idx, adp_name in self.adapter_configs.items():
            if not adp_name:
                continue
                
            # 로컬 경로 또는 캐시 경로 설정 (환경에 맞춰 수정 가능)
            path = f"./adapter/{adp_name}/adapter_model.bin"
            
            if os.path.exists(path):
                self.logger.info(f"==> [RAM 로드] 어댑터 {idx}번: {path}")
                # VRAM 절약을 위해 CPU에 로드
                self.adapter_cache[idx] = torch.load(path, map_location="cpu")
            else:
                self.logger.warning(f"⚠️ 어댑터 파일을 찾을 수 없음: {path}")

    def set_policy(self, policy):
        self.policy = policy

    def switch(self, adapter_index: int):
        """가중치 덮어쓰기를 통해 실시간으로 어댑터를 교체합니다."""
        if self.policy is None:
            self.logger.error("❌ 정책 모델이 로드되지 않았습니다.")
            return
            
        if adapter_index not in self.adapter_cache:
            self.logger.warning(f"⚠️ 캐시된 어댑터 없음 (ID: {adapter_index}). 베이스 상태로 유지됩니다.")
            return

        self.logger.info(f"🔄 어댑터 {adapter_index}번 주입 중 (Hot-swapping)...")
        
        # XVLA 가중치 주입 (strict=False로 베이스 모델 유지)
        state_dict = self.adapter_cache[adapter_index]
        self.policy.model.load_state_dict(state_dict, strict=False)
        
        self.logger.info(f"✅ 어댑터 {adapter_index}번 활성화 완료.")