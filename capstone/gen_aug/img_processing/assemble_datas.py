import os
from datasets import load_dataset, Image, Dataset
import pandas as pd
from tqdm import tqdm

def finalize_dataset(original_id, aug_img_dir, output_name):
    print(f"1. 원본 데이터셋 '{original_id}' 불러오는 중...")
    # 원본 데이터의 구조(Action, State, Episode 등)를 가져옵니다.
    ds_orig = load_dataset(original_id, split="train")

    # 2. 새로운 이미지 경로 리스트 생성
    # augmented_images 폴더에 있는 파일들을 순서대로 매칭합니다.
    aug_files = sorted([os.path.join(aug_img_dir, f) for f in os.listdir(aug_img_dir) if f.endswith('.jpg')])

    if len(ds_orig) != len(aug_files):
        print(f"주의: 원본 데이터({len(ds_orig)}개)와 합성 이미지({len(aug_files)}개)의 개수가 다릅니다!")
        # 개수가 다르면 작은 쪽에 맞춥니다.
        min_len = min(len(ds_orig), len(aug_files))
        ds_orig = ds_orig.select(range(min_len))
        aug_files = aug_files[:min_len]

    print("2. 이미지 교체 및 최종 데이터셋 구성 중...")

    # 원본 데이터셋을 Pandas로 변환하여 이미지 컬럼만 교체
    df = ds_orig.to_pandas()
    df['observation.image'] = aug_files # 경로를 새 이미지 경로로 업데이트

    # 다시 Hugging Face Dataset으로 변환
    final_ds = Dataset.from_pandas(df)

    # 문자열 경로를 실제 이미지 객체로 변환
    final_ds = final_ds.cast_column("observation.image", Image())

    print(f"3. 최종 파일 저장 중: {output_name}")
    final_ds.to_parquet(output_name)
    print("\\n모든 작업이 완료되었습니다! 이제 이 Parquet 파일을 학습에 사용하세요.")

if __name__ == "main":
    ORIGINAL_DATASET = "lerobot/pusht_image" # 원본 ID
    AUGMENTED_DIR = "augmented_images"       # 합성된 이미지가 있는 폴더
    FINAL_OUTPUT = "pusht_genaug_final.parquet"