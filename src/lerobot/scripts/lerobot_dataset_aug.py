#!/usr/bin/env python

"""
Unified Dataset Augmentation & Merge Script for Language Instructions
(SELF-CONTAINED: 외부 스크립트 의존성 없음)

사용 예시:
    # 인터랙티브 모드
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \
        --dataset_output_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket_AUG"

    # 지시어 파일 사용
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket" \
        --dataset_output_repo_id="roboseasy/Pick_up_a_sock_and_place_it_in_the_basket_AUG" \
        --instructions_file="instructions.txt"

    # 명령줄 직접 입력
    lerobot-dataset-aug \
        --dataset_load_repo_id="roboseasy/soarm_pick_and_place_socks" \
        --dataset_output_repo_id="roboseasy/soarm_pick_and_place_socks_aug" \
        --instructions "Pick up the sock and put it in the basket" \
        --instructions "Grab the sock and place it into the basket" \
        --instructions "Task the sock and put it into the basket" \
        --instructions "Lift the sock and drop it into the basket" \
        --instructions "Pick the sock and basket it" \
        --push_to_hub
"""

import argparse
import os
import sys
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi

def get_dataset_path(repo_id):
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

def duplicate_dataset(src_repo_id, dst_repo_id, single_task):
    """
    허깅페이스에서 직접 로드 → 전체 복사 → single_task만 변경
    """
    print(f"\n{'='*60}")
    print(f"Duplicating: {src_repo_id}")
    print(f"        To: {dst_repo_id}")
    print(f"      Task: {single_task}")
    print(f"{'='*60}\n")
    # 1. 원본 데이터셋 로드
    print("Step 1: Loading dataset from Hugging Face...")
    src_ds = LeRobotDataset(src_repo_id)
    src_root = Path(src_ds.root)
    print(f"  Original episodes: {src_ds.meta.total_episodes}")
    print(f"  Original frames: {src_ds.meta.total_frames}")
    print(f"  Root: {src_root}")
    # 2. 새 디렉토리 경로 설정
    dst_root = get_dataset_path(dst_repo_id)
    print(f"\nStep 2: Copying to {dst_root}...")
    if dst_root.exists():
        print("  Removing existing directory...")
        shutil.rmtree(dst_root)
    # 3. 전체 복사
    shutil.copytree(src_root, dst_root)
    print("  ✓ Copied all files (videos, data, meta)")
    # 4. single_task만 변경
    print(f"\nStep 3: Updating single_task...")
    info_path = dst_root / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    old_task = info.get("single_task", "N/A")
    info["single_task"] = single_task
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  Old: {old_task}")
    print(f"  New: {single_task}")
    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"{'='*60}")
    print(f"  Local: {dst_root}")
    print(f"{'='*60}\n")
    return dst_root

def merge_datasets(repo_ids, output_repo_id, push_to_hub=False):
    dataset_paths = [get_dataset_path(rid) for rid in repo_ids]
    output_path = get_dataset_path(output_repo_id)
    print(f"\n{'='*70}")
    print(f"Generic N-way Dataset Merging")
    print(f"{'='*70}")
    for i, (rid, p) in enumerate(zip(repo_ids, dataset_paths)):
        print(f"Repo {i}: {rid} -> {p}")
    print(f"Output:    {output_path}")
    print(f"{'='*70}\n")
    # Validate input datasets
    for p in dataset_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
    # Load metadata
    def load_info_json(dataset_path):
        info_path = dataset_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            return json.load(f)
    def load_tasks(dataset_path):
        tasks_path = dataset_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            df = pd.read_parquet(tasks_path)
            return list(df.index)
        return []
    infos = [load_info_json(p) for p in dataset_paths]
    tasks_list = [load_tasks(p) for p in dataset_paths]
    total_episodes_list = [info["total_episodes"] for info in infos]
    total_frames_list = [info["total_frames"] for info in infos]
    task_names = []
    for i, tasks in enumerate(tasks_list):
        if tasks:
            task_names.append(tasks[0])
        else:
            task_names.append(infos[i].get("single_task", f"Task {i+1}"))
    # Phase 1: Copy first dataset as base
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(dataset_paths[0], output_path)
    # Phase 2: Merge remaining datasets
    episode_offset = total_episodes_list[0]
    frame_offset = total_frames_list[0]
    video_file_offset = {}

    ##################################################################################
    # cameras = cameras = ["observation.images.camera1", "observation.images.camera2"]
   
    video_base_path = dataset_paths[0] / "videos"
    if video_base_path.exists():
        cameras = [
            p.name for p in video_base_path.iterdir() 
            if p.is_dir() and p.name.startswith("observation.images.")
        ]
    else:
        cameras = []
    ##################################################################################
    
    for cam in cameras:
        cam_dir = output_path / "videos" / cam / "chunk-000"
        video_file_offset[cam] = len(list(cam_dir.glob("file-*.mp4")))
    data_output_dir = output_path / "data" / "chunk-000"
    episodes_output_dir = output_path / "meta" / "episodes" / "chunk-000"
    data_file_counter = len(list(data_output_dir.glob("file-*.parquet")))
    ep_file_counter = len(list(episodes_output_dir.glob("file-*.parquet")))
    last_data_file = sorted(data_output_dir.glob("file-*.parquet"))[-1]
    last_index = pd.read_parquet(last_data_file)["index"].max()
    current_index = last_index + 1
    for i in range(1, len(dataset_paths)):
        # Data files
        data_dir = dataset_paths[i] / "data" / "chunk-000"
        data_files = sorted(data_dir.glob("file-*.parquet"))
        for data_file in data_files:
            df = pd.read_parquet(data_file)
            df["episode_index"] = df["episode_index"] + episode_offset
            df["task_index"] = i
            num_frames = len(df)
            df["index"] = range(current_index, current_index + num_frames)
            output_file = data_output_dir / f"file-{data_file_counter:03d}.parquet"
            df.to_parquet(output_file, index=False)
            data_file_counter += 1
            current_index += num_frames
        # Episodes files
        ep_dir = dataset_paths[i] / "meta" / "episodes" / "chunk-000"
        ep_files = sorted(ep_dir.glob("file-*.parquet"))
        for ep_file in ep_files:
            ep_df = pd.read_parquet(ep_file)
            ep_df["episode_index"] = ep_df["episode_index"] + episode_offset
            if "dataset_from_index" in ep_df.columns:
                ep_df["dataset_from_index"] = ep_df["dataset_from_index"] + frame_offset
            if "dataset_to_index" in ep_df.columns:
                ep_df["dataset_to_index"] = ep_df["dataset_to_index"] + frame_offset
            if "tasks" in ep_df.columns:
                ep_df["tasks"] = ep_df["tasks"].apply(lambda x: [task_names[i]] if isinstance(x, list) else [task_names[i]])
            for cam in cameras:
                file_index_col = f"videos/{cam}/file_index"
                to_timestamp_col = f"videos/{cam}/to_timestamp"
                if file_index_col in ep_df.columns:
                    ep_df[file_index_col] = ep_df[file_index_col] + video_file_offset[cam]
                if to_timestamp_col in ep_df.columns:
                    ep_df[to_timestamp_col] = ep_df[to_timestamp_col] - 0.01
            if "task_index" in ep_df.columns:
                ep_df = ep_df.drop(columns=["task_index"])
            output_file = episodes_output_dir / f"file-{ep_file_counter:03d}.parquet"
            ep_df.to_parquet(output_file, index=False)
            ep_file_counter += 1
        # Video files
        for cam in cameras:
            cam_dir = dataset_paths[i] / "videos" / cam / "chunk-000"
            output_cam_dir = output_path / "videos" / cam / "chunk-000"
            video_files = sorted(cam_dir.glob("file-*.mp4"))
            v_counter = video_file_offset[cam]
            for video_file in video_files:
                output_video = output_cam_dir / f"file-{v_counter:03d}.mp4"
                shutil.copy2(video_file, output_video)
                v_counter += 1
            video_file_offset[cam] = v_counter
        episode_offset += total_episodes_list[i]
        frame_offset += total_frames_list[i]
    # Phase 3: Update metadata
    info_output_path = output_path / "meta" / "info.json"
    with open(info_output_path, "r") as f:
        info_merged = json.load(f)
    info_merged["total_episodes"] = sum(total_episodes_list)
    info_merged["total_frames"] = sum(total_frames_list)
    info_merged["total_tasks"] = len(repo_ids)
    info_merged["splits"]["train"] = f"0:{sum(total_episodes_list)}"
    if "single_task" in info_merged:
        del info_merged["single_task"]
    with open(info_output_path, "w") as f:
        json.dump(info_merged, f, indent=4, ensure_ascii=False)
    # Create tasks.parquet
    tasks_parquet_path = output_path / "meta" / "tasks.parquet"
    tasks_df = pd.DataFrame({
        "task_index": list(range(len(task_names)))
    }, index=task_names)
    tasks_df.to_parquet(tasks_parquet_path)
    # Update stats.json
    stats_path = output_path / "meta" / "stats.json"
    all_data_files = sorted((output_path / "data" / "chunk-000").glob("file-*.parquet"))
    dfs = [pd.read_parquet(f) for f in all_data_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    stats = {}
    for column in merged_df.columns:

    ##################################################################################    
        # if column in ["observation.images.top", "observation.images.wrist"]:
        #     continue
    
        if column.startswith("observation.images."):
            continue
    ##################################################################################
    
        col_data = merged_df[column]
        if pd.api.types.is_numeric_dtype(col_data):
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
    if stats_path.exists():
        with open(stats_path, "r") as f:
            old_stats = json.load(f)

    ##################################################################################
        # if "observation.images.top" in old_stats:
        #     stats["observation.images.top"] = old_stats["observation.images.top"]
        # if "observation.images.wrist" in old_stats:
        #     stats["observation.images.wrist"] = old_stats["observation.images.wrist"]

        for key, value in old_stats.items():
            if key.startswith("observation.images."):
                stats[key] = value
    ##################################################################################

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    # Phase 4: Push to hub if requested
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
    parser = argparse.ArgumentParser(description="Augment dataset with multiple language instructions.")
    parser.add_argument("--dataset_load_repo_id", type=str, required=True, help="Source dataset repo_id (or path)")
    parser.add_argument("--dataset_output_repo_id", type=str, required=True, help="Output dataset repo_id (or path)")
    parser.add_argument("--instructions", type=str, action="append", help="Instruction string (can be used multiple times)")
    parser.add_argument("--instructions_file", type=str, help="Path to file with instructions (one per line)")
    parser.add_argument("--push_to_hub", action="store_true", help="If set, push merged dataset to HuggingFace hub")
    return parser.parse_args()

def interactive_instructions():
    print("새로운 지시어를 정해주세요 (엔터로 입력, esc 또는 빈 줄로 종료):")
    instructions = []
    try:
        while True:
            line = input("> ").strip()
            if not line:
                print("입력 종료.")
                break
            instructions.append(line)
            print("다음 지시어를 정해주세요 (계속하려면 엔터, 끝내려면 esc 또는 빈 줄):")
    except (KeyboardInterrupt, EOFError):
        print("\n입력 종료.")
    return instructions

def load_instructions(args):
    # 명령줄 입력
    if args.instructions:
        return args.instructions
    # 파일 입력
    if args.instructions_file:
        if not os.path.exists(args.instructions_file):
            print("지시어 파일이 없습니다.")
            sys.exit(1)
        with open(args.instructions_file, "r") as f:
            instructions = [line.strip() for line in f if line.strip()]
        if not instructions:
            print("지시어 파일에 유효한 지시어가 없습니다.")
            sys.exit(1)
        return instructions
    # 인터랙티브 입력
    return interactive_instructions()

def main():
    args = parse_args()
    instructions = load_instructions(args)
    if not instructions:
        print("지시어가 입력되지 않았습니다. 종료합니다.")
        sys.exit(1)
    # 1. instruction별 복제본 생성
    inst_repo_ids = []
    for idx, instruction in enumerate(tqdm(instructions, desc="지시어별 복제")):
        inst_repo_id = f"{args.dataset_output_repo_id}_inst{idx}"
        duplicate_dataset(
            src_repo_id=args.dataset_load_repo_id,
            dst_repo_id=inst_repo_id,
            single_task=instruction
        )
        inst_repo_ids.append(inst_repo_id)
    # 2. 병합 및 업로드
    merge_datasets(
        repo_ids=inst_repo_ids,
        output_repo_id=args.dataset_output_repo_id,
        push_to_hub=args.push_to_hub
    )
    # 3. 임시 복제본 자동 삭제(clean-up)
    for inst_repo_id in inst_repo_ids:
        inst_path = get_dataset_path(inst_repo_id)
        if inst_path.exists():
            try:
                shutil.rmtree(inst_path)
                print(f"임시 복제본 삭제 완료: {inst_path}")
            except Exception as e:
                print(f"임시 복제본 삭제 실패: {inst_path} ({e})")
    print("Augmentation and merge complete.")
    print("  Output repo_id:", args.dataset_output_repo_id)
    print("  Instructions:", instructions)
    if args.push_to_hub:
        print("  (허깅페이스 허브에 업로드 완료)")

if __name__ == "__main__":
    main()