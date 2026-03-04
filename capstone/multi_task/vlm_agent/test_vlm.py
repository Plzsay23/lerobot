import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# lerobot 카메라 파이프라인
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

def main():
    print("=== VLM 시각 인지 테스트 시작 ===")
    
    # 1. 카메라 초기화
    print("📷 카메라를 켭니다...")
    camera_config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    camera = OpenCVCamera(camera_config)
    camera.connect()
    print("✅ 카메라 연결 완료!")

    # 2. VLM 로드 (Orin NX 최적화 적용)
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    print(f"🧠 VLM({model_id})을 메모리에 올리는 중... (최초 1회 수십 초 소요)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    vlm = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" # Orin NX 가속
    ).to(device)
    vlm.eval()
    print("✅ VLM 로드 완료!\n")

    try:
        while True:
            print("="*60)
            user_question = input(">>> ")
            
            if user_question.lower() == 'q':
                break
            if not user_question.strip():
                continue

            # 카메라 캡처
            start_time = time.time()
            image_array = camera.read()
            image_pil = Image.fromarray(image_array)

            # VLM 프롬프트 구성
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_question}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image_pil], return_tensors="pt").to(device)

            print("Thinking...")
            
            # 텍스트 생성
            with torch.no_grad():
                generated_ids = vlm.generate(
                    **inputs,
                    max_new_tokens=100, # 답변 최대 길이
                    do_sample=True,     # 자연스러운 답변을 위해 허용
                    temperature=0.7     # 창의성 조절
                )
            
            # 결과 디코딩 및 출력
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            end_time = time.time()
            
            print(f"\n🤖 [Answer] (inference time: {end_time - start_time:.2f}s)")
            print(f">> {output_text}\n")

    finally:
        print("🔌 카메라를 끄고 프로그램을 종료합니다.")
        if camera.is_connected:
            camera.disconnect()

if __name__ == "__main__":
    main()