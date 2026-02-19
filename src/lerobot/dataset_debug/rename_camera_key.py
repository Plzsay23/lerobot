import os
import json
import pyarrow.parquet as pq
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def rename_dataset_camera():
    # 1. 환경변수 읽기
    hf_user = os.getenv("HF_USER")
    data_repo = os.getenv("DATA_REPO")
    result_repo = os.getenv("RESULT")
    
    if not all([hf_user, data_repo, result_repo]):
        print("❌ 환경변수(HF_USER, DATA_REPO, RESULT)를 확인해주세요.")
        return

    src_repo_id = f"{hf_user}/{data_repo}"
    dst_repo_id = f"{hf_user}/{result_repo}"
    
    print(f"🔄 다운로드 및 로드 시작: {src_repo_id}")
    
    # 2. 데이터셋 다운로드 및 로드
    # LeRobotDataset은 인스턴스화 시 로컬 캐시에 없으면 자동으로 다운로드합니다.
    dataset = LeRobotDataset(src_repo_id)
    root_path = Path(dataset.root)

    old_key = "observation.images.top"
    new_key = "observation.images.front"

    # 3. 메타데이터 (info.json) 수정
    info_path = root_path / "meta/info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    if old_key in info["features"]:
        info["features"][new_key] = info["features"].pop(old_key)
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"✅ info.json 수정 완료: {old_key} -> {new_key}")
    else:
        print(f"⚠️ {old_key}를 찾을 수 없습니다. 이미 수정되었거나 키가 다를 수 있습니다.")
        return

    # 4. 데이터 파일 (Parquet) 수정
    # data/ 폴더 내의 모든 파일을 순회하며 컬럼명을 변경합니다.
    parquet_files = list((root_path / "data").rglob("*.parquet"))
    for p_file in parquet_files:
        table = pq.read_table(p_file)
        if old_key in table.column_names:
            new_column_names = [new_key if col == old_key else col for col in table.column_names]
            table = table.rename_columns(new_column_names)
            pq.write_table(table, p_file)
    print(f"✅ 모든 Parquet 파일 ({len(parquet_files)}개) 수정 완료")

    # 5. 비디오 폴더 이름 변경
    # info.json 구조에 따라 비디오가 저장된 물리적 폴더명을 변경합니다.
    old_video_dir = root_path / f"videos/{old_key}"
    new_video_dir = root_path / f"videos/{new_key}"
    if old_video_dir.exists():
        old_video_dir.rename(new_video_dir)
        print(f"✅ 비디오 디렉토리 이름 변경 완료")

    # 6. 새로운 이름으로 업로드
    # 데이터셋의 repo_id를 새로운 이름으로 교체한 뒤 push_to_hub를 호출합니다.
    dataset.repo_id = dst_repo_id
    print(f"🚀 업로드 시작: {dst_repo_id}")
    dataset.push_to_hub()
    print("✨ 모든 작업이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    rename_dataset_camera()