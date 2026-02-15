import os
import torch
import shutil
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from huggingface_hub import HfApi, whoami

def get_current_username():
    try:
        return whoami()['name']
    except Exception:
        print("❌ 오류: 허깅페이스 로그인이 필요합니다.")
        exit(1)

def main():
    # 환경 변수 및 유저 정보 설정
    username = get_current_username()
    model_name = os.getenv("MODEL_NAME") # 예: pap_black_xvla
    
    src_repo = f"{username}/{model_name}"
    dest_repo = f"{username}/{model_name}_adapter"
    tmp_dir = "./tmp_xvla_adapter"
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"==> 1. {src_repo}에서 XVLA 모델 로드 중...")
    # XVLAPolicy.from_pretrained로 전체 가중치 로드 
    policy = XVLAPolicy.from_pretrained(src_repo)
    
    print("==> 2. XVLA 어댑터 가중치 추출 중 (키워드 매칭)...")
    adapter_state_dict = {}
    
    # XVLA에서 학습 가능한 핵심 레이어 키워드 
    # - transformer: 정책 결정 헤드
    # - soft_prompt_hub: 소프트 프롬프트
    # - action_space: 액션 관련 레이어
    target_keywords = ["transformer", "soft_prompt_hub", "action_space"]
    
    found_keys = 0
    for name, param in policy.model.named_parameters():
        if any(key in name for key in target_keywords):
            # 'model.' 접두사가 붙어있을 수 있으므로 가중치 이름 유지
            adapter_state_dict[name] = param.cpu()
            found_keys += 1
    
    if found_keys == 0:
        print("❌ 오류: 추출할 가중치를 찾지 못했습니다. 키워드를 확인하세요.")
        return

    print(f"   (총 {found_keys}개의 레이어 추출 완료)")

    # 가중치 및 설정 저장
    torch.save(adapter_state_dict, os.path.join(tmp_dir, "adapter_model.bin"))
    policy.config.save_pretrained(tmp_dir)
    
    print(f"==> 3. {dest_repo}로 업로드 중...")
    api = HfApi()
    api.create_repo(repo_id=dest_repo, exist_ok=True)
    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=dest_repo,
        repo_type="model",
    )
    
    shutil.rmtree(tmp_dir)
    print(f"✅ 완료! 주소: https://huggingface.co/{dest_repo}")

if __name__ == "__main__":
    main()