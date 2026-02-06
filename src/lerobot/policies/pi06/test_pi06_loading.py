import sys
from unittest.mock import MagicMock

# ==========================================================
# [최종 패치 v4] "옆집 모델들은 제발 조용히 해" 에디션
# ----------------------------------------------------------
# lerobot/__init__.py가 불러오려는 "다른 모델 파일들"만 콕 집어서 미리 가짜로 채워둡니다.
# 이렇게 하면 폴더 구조 에러 없이 통과됩니다.
# ----------------------------------------------------------

# 1. Groot 관련 파일 차단 (상세 경로 지정)
sys.modules["lerobot.policies.groot.configuration_groot"] = MagicMock()
sys.modules["lerobot.policies.groot.modeling_groot"] = MagicMock()

# 2. XVLA 관련 파일 차단
sys.modules["lerobot.policies.xvla.configuration_xvla"] = MagicMock()
sys.modules["lerobot.policies.xvla.processor_xvla"] = MagicMock()
sys.modules["lerobot.policies.xvla.modeling_xvla"] = MagicMock()

# 3. 로봇 & 텔레오퍼레이터 차단
sys.modules["lerobot.robots.so101_follower"] = MagicMock()
sys.modules["lerobot.robots.bi_so101_follower"] = MagicMock()
sys.modules["lerobot.teleoperators.so101_leader"] = MagicMock()
sys.modules["lerobot.teleoperators.bi_so101_leader"] = MagicMock()

# 4. 데이터셋 팩토리 차단
sys.modules["lerobot.datasets.factory"] = MagicMock()
# ==========================================================

import torch
from lerobot.policies.pi06.configuration_pi06 import PI06Config
from lerobot.policies.pi06.modeling_pi06 import PI06Policy

# [핵심] 8GB VRAM을 위한 '가짜' Tiny 설정 주입 함수
def mock_get_gemma_config(variant: str):
    from lerobot.policies.pi06.modeling_pi06 import GemmaConfig
    # 무조건 아주 작은 모델 스펙을 반환하게 만듦
    print(f"    [Test Mode] '{variant}' 요청을 감지했습니다 -> Tiny 사이즈로 축소하여 반환합니다.")
    return GemmaConfig(
        width=128,       # 원래 3072 -> 128로 축소
        depth=2,         # 원래 34 -> 2로 축소
        mlp_dim=512,     
        num_heads=4,     
        num_kv_heads=1,
        head_dim=32,
    )

def test_pi06_instantiation():
    print(">>> 1. [Test] 8GB VRAM 안전 모드로 설정(Config) 초기화 중...")
    
    # 1. modeling 파일의 함수를 우리가 만든 가짜 함수로 바꿔치기 (Monkey Patching)
    import lerobot.policies.pi06.modeling_pi06 as modeling_module
    original_get_config = modeling_module.get_gemma_config
    modeling_module.get_gemma_config = mock_get_gemma_config
    
    try:
        config = PI06Config(
            paligemma_variant="gemma_3_4b",    # 이름은 그대로 유지 (검증 로직 통과 위해)
            action_expert_variant="gemma_860m",
            image_resolution=(224, 224),
            # text_hidden_size 삭제 완료
        )

        print("\n>>> 2. 모델 로드 중 (Tiny 버전)...")
        # 이제 GPU에 올려도 안전합니다 (약 100MB도 안 됨)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    - Device: {device}")
        
        policy = PI06Policy(config)
        policy.to(device)
        print("    - 모델 로드 성공! (VRAM OOM 회피)")

        print("\n>>> 3. 더미 데이터(Forward) 테스트...")
        batch_size = 2
        # Tiny 모델 스펙에 맞춘 더미 데이터
        dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_img_masks = torch.ones(batch_size, dtype=torch.bool).to(device)
        dummy_tokens = torch.randint(0, 1000, (batch_size, 16)).to(device)
        dummy_masks = torch.ones(batch_size, 16, dtype=torch.bool).to(device)
        dummy_actions = torch.randn(batch_size, config.chunk_size, config.max_action_dim).to(device)

        loss = policy.model.forward(
            images=[dummy_images], 
            img_masks=[dummy_img_masks], 
            tokens=dummy_tokens, 
            masks=dummy_masks, 
            actions=dummy_actions
        )
        
        print(f"    - Forward 성공! Loss shape: {loss.shape}")
        print("    - Loss 값(mean):", loss.mean().item())
        print("\n>>> [성공] 코드 로직에 이상이 없습니다. 실제 학습은 고사양 서버에서 하세요!")

    except Exception as e:
        print(f"    [!!!] 실패: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 테스트 끝나면 원래대로 돌려놓기
        modeling_module.get_gemma_config = original_get_config

if __name__ == "__main__":
    test_pi06_instantiation()