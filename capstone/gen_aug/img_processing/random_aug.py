import os
import random
from PIL import Image
from tqdm import tqdm

def apply_genaug(img_dir, bg_dir, output_dir):
    # 1. 저장할 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")

    # 2. 파일 목록 읽기
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    bg_files = [f for f in os.listdir(bg_dir) if f.endswith(('.jpg', '.png'))]

    if not bg_files:
        print("에러: backgrounds 폴더에 배경 사진을 넣어주세요!")
        return

    print(f"총 {len(img_files)}개 이미지에 배경 합성(GenAug) 적용 중...")

    # 3. 이미지 하나씩 작업
    for file_name in tqdm(img_files):
        # 로봇 이미지 로드 (투명도를 다루기 위해 RGBA 모드로 변환)
        robot_img = Image.open(os.path.join(img_dir, file_name)).convert("RGBA")

        # 배경 사진 중 하나를 무작위로 선택하여 로봇 이미지 크기에 맞춤
        bg_path = os.path.join(bg_dir, random.choice(bg_files))
        bg_img = Image.open(bg_path).convert("RGBA").resize(robot_img.size)

        # 배경 제거 (흰색 부분을 투명하게 만들기)
        datas = robot_img.getdata()
        new_data = []
        for item in datas:
            # RGB 값이 모두 240 이상이면 흰색으로 판단 (조절 가능)
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                new_data.append((255, 255, 255, 0)) # 투명하게 변경
            else:
                new_data.append(item)
        robot_img.putdata(new_data)

        # 배경 이미지 위에 로봇 이미지를 얹음
        combined = Image.alpha_composite(bg_img, robot_img)

        # 다시 일반 사진 형태(RGB)로 바꿔서 저장
        combined.convert("RGB").save(os.path.join(output_dir, file_name), "JPEG")

    print(f"\\n작업 완료 합성된 사진들이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "main":
    # 폴더 설정: 원본이미지, 배경이미지, 결과물저장소
    apply_genaug("extracted_images", "backgrounds", "augmented_images")
