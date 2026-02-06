import os
import pandas as pd
from datasets import Dataset, Image
from tqdm import tqdm

def run_conversion():
    img_dir = "extracted_images"
    output_file = "practice.parquet"

    # 1. 폴더 내 jpg 파일 목록 가져오기 (이름순 정렬)
    print("파일 목록 읽는 중...")
    file_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    # 연습용이므로 상위 100개만 테스트해보고 싶다면 아래 주석을 해제하세요.
    # file_names = file_names[:100]

    data_rows = []
    for file_name in tqdm(file_names, desc="데이터 구성 중"):
        full_path = os.path.join(os.getcwd(), img_dir, file_name)

        # LeRobot 형식에 맞춰 'observation.image'라는 이름을 사용합니다.
        data_rows.append({
            "observation.image": full_path,
            "filename": file_name
        })

    # 2. 데이터프레임 생성 및 Dataset 변환
    df = pd.DataFrame(data_rows)
    ds = Dataset.from_pandas(df)

    # 3. 중요: 단순 문자열 경로를 '이미지 객체'로 인식하도록 변환
    ds = ds.cast_column("observation.image", Image())

    # 4. Parquet 파일로 저장
    print(f"\\n'{output_file}' 생성 시작...")
    ds.to_parquet(output_file)
    print(f"완료! {len(file_names)}개의 이미지가 Parquet으로 변환되었습니다.")

if __name__ == "main":
    run_conversion()
