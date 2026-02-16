import os
import torch

class MultiAdapterManager:
    def __init__(self, logger):
        self.logger = logger
        self.policy = None
        self.adapter_cache = {}
        # 서버 초기화 시점에 어댑터 로드 함수 호출
        self._load_all_adapters_to_vram()

    def _load_all_adapters_to_vram(self):
        """어댑터 가중치를 시작부터 VRAM(GPU)에 상주시킵니다."""
        # 사용할 디바이스 설정 (일반적으로 'cuda')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        configs = {
            1: os.getenv("ADP1"),
            2: os.getenv("ADP2"),
        }
        
        for idx, name in configs.items():
            if not name: continue
            path = f"./adapter/{name}/adapter_model.bin"
            if os.path.exists(path):
                # [수정] map_location을 GPU로 설정하여 즉시 VRAM 점유
                self.logger.info(f"🚀 어댑터 {idx}번 VRAM 상주 시작...")
                self.adapter_cache[idx] = torch.load(path, map_location=device)
                self.logger.info(f"✅ 어댑터 {idx}번 VRAM 로드 완료")

    def set_policy(self, policy):
        self.policy = policy

    def switch(self, adapter_index: int):
        if self.policy is None or adapter_index not in self.adapter_cache:
            return

        # VRAM 내 가중치 복사 (Bus 전송이 없어 매우 빠름)
        state_dict = self.adapter_cache[adapter_index]
        self.policy.model.load_state_dict(state_dict, strict=False)

    def verify_adapter_delta(self):
    # 어댑터가 적용된 파라미터와 베이스 파라미터의 차이 확인
    # 만약 모든 차이값이 0에 가깝다면 어댑터가 빈 껍데기일 확률이 높습니다.
        for name, param in self.policy.named_parameters():
            if "adapter" in name or "lora" in name:
                weight_max = param.abs().max().item()
                weight_mean = param.abs().mean().item()
                self.logger.info(f"📊 [ADAPTER-CHECK] {name} | Max: {weight_max:.6f} | Mean: {weight_mean:.6f}")