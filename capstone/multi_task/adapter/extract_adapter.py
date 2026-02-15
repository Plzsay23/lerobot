import os
import torch
from peft import PeftModel
from lerobot.policies.factory import make_policy
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

    print(f"[{source_repo}]에서 모델 로드 중...")
    
    # 1. LeRobot Factory를 통해 모델 로드
    # 이 과정에서 모델은 로컬 캐시에 다운로드됩니다.
    policy = make_policy(repo_id=source_repo)
    
    # 2. 어댑터 객체 식별
    # SmolVLA나 xVLA 등 모델 구조에 따라 어댑터가 위치한 속성을 찾습니다.
    model_to_extract = None
    if hasattr(policy, "model") and isinstance(policy.model, PeftModel):
        model_to_extract = policy.model
    elif hasattr(policy, "vlm_with_expert") and hasattr(policy.vlm_with_expert, "vlm"):
        if isinstance(policy.vlm_with_expert.vlm, PeftModel):
            model_to_extract = policy.vlm_with_expert.vlm

    if model_to_extract is None:
        print("에러: 해당 모델에서 PeftModel(어댑터) 구조를 찾을 수 없습니다.")
        return

    # 3. 어댑터만 로컬 임시 디렉토리에 저장
    # 이 메소드는 베이스 모델 가중치를 제외하고 오직 어댑터 파일들만 생성합니다.
    print(f"어댑터 추출 중: {tmp_save_dir}")
    model_to_extract.save_pretrained(tmp_save_dir)

    # 4. 허깅페이스에 새로운 레포로 업로드
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