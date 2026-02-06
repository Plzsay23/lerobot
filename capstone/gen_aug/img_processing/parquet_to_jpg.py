import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def extract_images_from_lerobot(dataset_name, output_dir="extracted_images"):
    # 1. 데이터셋 로드 (로컬에 없으면 자동으로 다운로드됨)
    print(f"데이터셋 '{dataset_name}' 로딩 중...")
    ds = load_dataset(dataset_name, split="train")  # 기본적으로 'train' 스플릿 사용

    # 2. 저장할 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")

    # 3. 이미지 추출 및 저장

    print("이미지 추출 시작...")
    for i, example in enumerate(tqdm(ds)):
        image = example.get('observation.image')

    if image is None:
        if i == 0:
            print(f"사용 가능한 컬럼들: {example.keys()}")
        continue

    # 파일명 설정 (예: frame_000001.jpg)
    file_name = f"frame_{i:06d}.jpg"
    file_path = os.path.join(output_dir, file_name)

    # 이미지가 PIL Image 객체이므로 바로 저장 가능
    if isinstance(image, Image.Image):
        image.save(file_path, "JPEG")
    else:
        # 만약 바이너리 형태라면 변환이 필요할 수 있음
        image.save(file_path)

    print(f"\\n작업 완료! {i + 1}개의 이미지가 '{output_dir}'에 저장되었습니다.")

if __name__ == "main":
    # 사용하고자 하는 데이터셋 주소 입력
    DATASET_ID = "lerobot/pusht_image"
    extract_images_from_lerobot(DATASET_ID)
