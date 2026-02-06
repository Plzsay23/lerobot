#!/usr/bin/env python

"""
Robust Dataset Merge Script for Multiple Datasets (task_index 누적/연속 보장)
- 서로 다른 데이터셋을 병합
- task_index가 절대 겹치지 않도록 누적 offset 방식으로 부여
- 기존 병합 데이터셋에 추가 병합 시에도 task_index 연속성 보장
- HuggingFace 허브 업로드 옵션 지원
- lerobot_dataset_aug.py의 robust한 파일/메타데이터 처리 방식 참고

python simeple_merge.py \
  --dataset_repo_ids roboseasy/soarm_pick_and_place_blue_pen roboseasy/soarm_pick_and_place_red_pen \
  --output_repo_id roboseasy/soarm_pick_and_place_blue_red_merged \
  --push_to_hub

python src/lerobot/scripts/simeple_merge.py \
    --dataset_repo_ids roboseasy/soarm_pick_and_place_blue_pen roboseasy/soarm_pick_and_place_red_pen \
    --output_repo_id roboseasy/soarm_pick_and_place_blue_red_merged \
    --push_to_hub

python simeple_merge.py \
    --dataset_repo_ids roboseasy/soarm_pick_and_place_blue_red_merged roboseasy/soarm_pick_and_place_socks \
    --output_repo_id roboseasy/soarm_pick_and_place_pens_socks_merged \
    --push_to_hub


"""

import argparse
import os
import sys
import shutil
import json
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

def get_dataset_path(repo_id):
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

def get_last_task_index(output_path):
    tasks_parquet = output_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        df = pd.read_parquet(tasks_parquet)
        if len(df) > 0:
            return int(df["task_index"].max())
    return -1

def get_task_names(output_path):
    tasks_parquet = output_path / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        df = pd.read_parquet(tasks_parquet)
        return list(df.index)
    return []

def robust_merge_datasets(repo_ids, output_repo_id, push_to_hub=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_path = get_dataset_path(output_repo_id)
    output_exists = output_path.exists()
    if output_exists:
        print(f"[INFO] Output dataset {output_repo_id} already exists. Will append new datasets.")
        last_task_index = get_last_task_index(output_path)
        task_names_accum = get_task_names(output_path)
        total_episodes_accum = 0
        total_frames_accum = 0
        info_path = output_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
                total_episodes_accum = info.get("total_episodes", 0)
                total_frames_accum = info.get("total_frames", 0)
    else:
        last_task_index = -1
        task_names_accum = []
        total_episodes_accum = 0
        total_frames_accum = 0

    # 병합할 데이터셋 순회
    for rid in repo_ids:
        print(f"\n{'='*60}\nMerging: {rid}\n{'='*60}")
        # 1. 데이터셋 로컬에 없으면 다운로드
        ds = LeRobotDataset(rid)
        src_root = Path(ds.root)
        print(f"  Episodes: {ds.meta.total_episodes}, Frames: {ds.meta.total_frames}")
        # 2. task_name 추출
        tasks_parquet = src_root / "meta" / "tasks.parquet"
        if tasks_parquet.exists():
            task_names = list(pd.read_parquet(tasks_parquet).index)
        else:
            info_path = src_root / "meta" / "info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            task_names = [info.get("single_task", rid)]
        # 3. task_index offset 계산
        task_offset = last_task_index + 1
        print(f"  Assigning task_index offset: {task_offset}")
        # 4. output_path가 없으면 첫 데이터셋을 통째로 복사
        if not output_exists:
            print(f"  Copying base dataset to {output_path}")
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(src_root, output_path)
            # task_index 0으로 덮어쓰기
            # (첫 데이터셋은 항상 offset=0)
            data_dir = output_path / "data" / "chunk-000"
            for data_file in tqdm(sorted(data_dir.glob("file-*.parquet")), desc="  [Base] task_index relabel"):
                df = pd.read_parquet(data_file)
                df["task_index"] = 0
                df.to_parquet(data_file, index=False)
            # meta/episodes도 task_index 0으로
            ep_dir = output_path / "meta" / "episodes" / "chunk-000"
            for ep_file in sorted(ep_dir.glob("file-*.parquet")):
                ep_df = pd.read_parquet(ep_file)
                if "task_index" in ep_df.columns:
                    ep_df["task_index"] = 0
                    ep_df.to_parquet(ep_file, index=False)
            # tasks.parquet 생성
            tasks_df = pd.DataFrame({"task_index": [0]}, index=task_names)
            tasks_df.to_parquet(output_path / "meta" / "tasks.parquet")
            last_task_index = 0
            task_names_accum = task_names
            total_episodes_accum = ds.meta.total_episodes
            total_frames_accum = ds.meta.total_frames
            output_exists = True
            continue
        # 5. 이후 데이터셋은 offset 적용해서 병합
        # 데이터 parquet
        data_dir = src_root / "data" / "chunk-000"
        out_data_dir = output_path / "data" / "chunk-000"
        data_file_counter = len(list(out_data_dir.glob("file-*.parquet")))
        last_index = 0
        if data_file_counter > 0:
            last_data_file = sorted(out_data_dir.glob("file-*.parquet"))[-1]
            last_index = pd.read_parquet(last_data_file)["index"].max() + 1
        for data_file in tqdm(sorted(data_dir.glob("file-*.parquet")), desc="  [Data] task_index relabel & append"):
            df = pd.read_parquet(data_file)
            df["task_index"] = df["task_index"].apply(lambda x: x + task_offset)
            num_frames = len(df)
            df["index"] = range(last_index, last_index + num_frames)
            out_file = out_data_dir / f"file-{data_file_counter:03d}.parquet"
            df.to_parquet(out_file, index=False)
            data_file_counter += 1
            last_index += num_frames
        # meta/episodes parquet
        ep_dir = src_root / "meta" / "episodes" / "chunk-000"
        out_ep_dir = output_path / "meta" / "episodes" / "chunk-000"
        ep_file_counter = len(list(out_ep_dir.glob("file-*.parquet")))
        for ep_file in sorted(ep_dir.glob("file-*.parquet")):
            ep_df = pd.read_parquet(ep_file)
            if "task_index" in ep_df.columns:
                ep_df["task_index"] = ep_df["task_index"].apply(lambda x: x + task_offset)
            out_file = out_ep_dir / f"file-{ep_file_counter:03d}.parquet"
            ep_df.to_parquet(out_file, index=False)
            ep_file_counter += 1
        # videos 복사 (덮어쓰기 방지)
        for cam in ["observation.images.camera1", "observation.images.camera2"]:
            src_cam_dir = src_root / "videos" / cam / "chunk-000"
            out_cam_dir = output_path / "videos" / cam / "chunk-000"
            v_counter = len(list(out_cam_dir.glob("file-*.mp4")))
            for video_file in sorted(src_cam_dir.glob("file-*.mp4")):
                out_video = out_cam_dir / f"file-{v_counter:03d}.mp4"
                shutil.copy2(video_file, out_video)
                v_counter += 1
        # task_names 누적
        task_names_accum.extend(task_names)
        last_task_index += len(task_names)
        total_episodes_accum += ds.meta.total_episodes
        total_frames_accum += ds.meta.total_frames

    # 메타데이터 갱신
    info_path = output_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    info["total_episodes"] = total_episodes_accum
    info["total_frames"] = total_frames_accum
    info["total_tasks"] = len(task_names_accum)
    info["splits"]["train"] = f"0:{total_episodes_accum}"
    if "single_task" in info:
        del info["single_task"]
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    # tasks.parquet 갱신
    tasks_df = pd.DataFrame({
        "task_index": list(range(len(task_names_accum)))
    }, index=task_names_accum)
    tasks_df.to_parquet(output_path / "meta" / "tasks.parquet")
    # stats.json 갱신 (모든 key superset 포함)
    stats_path = output_path / "meta" / "stats.json"
    all_data_files = sorted((output_path / "data" / "chunk-000").glob("file-*.parquet"))
    dfs = [pd.read_parquet(f) for f in all_data_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    # 병합 대상 데이터셋들의 stats.json key superset 수집
    all_stats_keys = set(merged_df.columns)
    # 기존 stats.json에서 key도 모두 포함
    if stats_path.exists():
        with open(stats_path, "r") as f:
            old_stats = json.load(f)
        all_stats_keys.update(old_stats.keys())
    stats = {}
    for column in all_stats_keys:
        if column in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[column]):
            col_data = merged_df[column]
            stats[column] = {
                "min": [float(col_data.min())],
                "max": [float(col_data.max())],
                "mean": [float(col_data.mean())],
                "std": [float(col_data.std())],
                "count": [len(col_data)],
                "q01": [float(col_data.quantile(0.01))],
                "q10": [float(col_data.quantile(0.10))],
                "q50": [float(col_data.quantile(0.50))],
                "q90": [float(col_data.quantile(0.90))],
                "q99": [float(col_data.quantile(0.99))]
            }
        elif stats_path.exists() and column in old_stats:
            stats[column] = old_stats[column]
        else:
            # 없는 경우 빈 값으로라도 포함
            stats[column] = {}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    # 허브 업로드
    if push_to_hub:
        api = HfApi()
        try:
            api.create_repo(repo_id=output_repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"Repository creation: {e}")
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=output_repo_id,
            repo_type="dataset"
        )
        try:
            api.create_tag(
                repo_id=output_repo_id,
                tag="v3.0",
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Warning: Could not create tag (may already exist): {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Robustly merge multiple datasets with unique task_index assignment.")
    parser.add_argument("--dataset_repo_ids", type=str, required=True, nargs='+', help="List of dataset repo_ids to merge")
    parser.add_argument("--output_repo_id", type=str, required=True, help="Output dataset repo_id")
    parser.add_argument("--push_to_hub", action="store_true", help="Push merged dataset to HuggingFace hub")
    return parser.parse_args()

def main():
    args = parse_args()
    robust_merge_datasets(args.dataset_repo_ids, args.output_repo_id, args.push_to_hub)
    print("Dataset robust merge complete.")
    print(f"Output repo_id: {args.output_repo_id}")

if __name__ == "__main__":
    main()