import os
import torch
from peft import PeftModel
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from huggingface_hub import HfApi

def main():
    # 환경 변수 읽기
    hf_user = os.getenv("HF_USER")
    model_name = os.getenv("MODEL_NAME")
    
    if not hf_user or not model_name:
        print("에러: HF_USER 또는 MODEL_NAME 환경 변수가 설정되지 않았습니다.")
        return

    source_repo = f"{hf_user}/{model_name}"
    target_repo = f"{hf_user}/{model_name}_adapter"
    tmp_save_dir = "temp_adapter_extract"

    print(f"[{source_repo}]에서 XVLA 모델 로드 중...")
    
    try:
        # lerobot의 방식대로 from_pretrained를 사용하여 로드 
        policy = XVLAPolicy.from_pretrained(source_repo)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # XVLA 모델 구조에서 PeftModel(어댑터) 위치 찾기 
    # XVLA는 내부적으로 model.vlm 등을 포함함 
    model_to_extract = None
    
    # XVLA 구조 탐색: policy.model.vlm이 보통 PeftModel임 
    if hasattr(policy, "model") and hasattr(policy.model, "vlm"):
        if isinstance(policy.model.vlm, PeftModel):
            model_to_extract = policy.model.vlm
    
    # 찾지 못했을 경우 전체 구조에서 PeftModel 검색
    if model_to_extract is None:
        for module in policy.modules():
            if isinstance(module, PeftModel):
                model_to_extract = module
                break

    if model_to_extract is None:
        print("에러: 이 모델에서 추출할 수 있는 Peft 어댑터를 찾지 못했습니다.")
        print("주의: 학습 시 --peft.method=LORA 옵션을 사용했는지 확인하세요.")
        return

    # 어댑터 파일만 추출
    print(f"어댑터 추출 중: {tmp_save_dir}")
    model_to_extract.save_pretrained(tmp_save_dir)

    # 허깅페이스 업로드
    print(f"[{target_repo}]로 업로드 중...")
    api = HfApi()
    api.create_repo(repo_id=target_repo, exist_ok=True)
    api.upload_folder(
        folder_path=tmp_save_dir,
        repo_id=target_repo,
        repo_type="model"
    )
    
    print(f"성공! 어댑터가 {target_repo}에 업로드되었습니다.")

if __name__ == "__main__":
    main()