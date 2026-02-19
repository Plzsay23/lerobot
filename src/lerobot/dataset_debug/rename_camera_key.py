import os
import json
from pathlib import Path
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def fix_stats_and_push():
    hf_user = os.getenv("HF_USER")
    data_repo = os.getenv("DATA_REPO")
    result_repo = os.getenv("RESULT")
    
    src_repo_id = f"{hf_user}/{data_repo}"
    dst_repo_id = f"{hf_user}/{result_repo}"
    
    # 데이터셋 로드 (수정된 로컬 캐시가 있다면 그것을 사용)
    dataset = LeRobotDataset(src_repo_id)
    root_path = Path(dataset.root)

    old_key = "observation.images.top"
    new_key = "observation.images.front"

    # 1. stats.json 수정 (가장 중요한 부분)
    stats_path = root_path / "meta/stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        if old_key in stats:
            stats[new_key] = stats.pop(old_key)
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"✅ stats.json 수정 완료: {old_key} -> {new_key}")
        else:
            print("⚠️ stats.json에 해당 키가 없거나 이미 수정되었습니다.")
    
    # 2. info.json 및 Parquet 재검토 (혹시 누락된 경우 대비)
    info_path = root_path / "meta/info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    if old_key in info["features"]:
        info["features"][new_key] = info["features"].pop(old_key)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

    # 3. 새로운 리포지토리로 다시 업로드
    dataset.repo_id = dst_repo_id
    dataset.push_to_hub()
    print(f"🚀 {dst_repo_id}로 수정된 통계 데이터와 함께 재업로드 완료!")

if __name__ == "__main__":
    fix_stats_and_push()