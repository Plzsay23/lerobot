# manager_agent.py
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class ManagerAgent:
    def __init__(self, available_models, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        self.models = available_models
        print(f"🔄 [System] 오리지널 VLM 백본 로드 중... ({model_id})")
        print("   (최초 실행 시 모델 다운로드에 시간이 약간 소요될 수 있습니다)")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Hugging Face에서 원본 VLM 직접 로드 (외계어 방지 및 bfloat16 정밀도 적용)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.vlm = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # 메모리 절약 및 출력 안정화
        ).to(self.device)
        self.vlm.eval()
        
        print("✅ [System] VLM 에이전트 준비 완료!")

    def observe_and_think(self, image_array, user_goal):
        """
        실제 카메라 이미지(NumPy)와 텍스트 명령을 받아 VLM으로 추론
        """
        print("🧠 [VLM 추론 중] 카메라 이미지와 명령어를 분석합니다...")
        
        # 이미지를 PIL 형식으로 변환 (OpenCVCamera는 이미 RGB 형태로 반환함)
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        image_pil = Image.fromarray(image_array)

        # VLM에게 내릴 프롬프트 작성 (포맷을 더 명확하게 줄바꿈으로 변경)
        system_prompt = (
            "You are a smart robotic task manager. Based on the user's instruction and the given image, "
            "decide which task to execute.\n"
            "Available tasks are: 'pick_place' and 'cleaning'.\n"
            "You MUST output your decision in the exact following format:\n"
            "Task: [task_name]\n"
            "Target: [object_name]\n"
            "Reason: [your reason]"
        )
        user_prompt = f"Instruction: {user_goal}"

        # Chat Template 적용
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image_pil], return_tensors="pt").to(self.device)

        # VLM 추론 실행 (do_sample=False로 설정하여 헛소리를 방지하고 일관된 답변 유도)
        with torch.no_grad():
            generated_ids = self.vlm.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        # 결과 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        print(f"🤖 [VLM Raw Output]\n{output_text}\n" + "-"*30)

        # 출력 텍스트 파싱 (줄바꿈 단위로 파싱)
        try:
            # 출력된 텍스트에서 각 항목 추출
            lines = output_text.split('\n')
            task_part = [line for line in lines if "Task:" in line][0].split("Task:")[1].strip()
            target_part = [line for line in lines if "Target:" in line][0].split("Target:")[1].strip()
            reason_part = [line for line in lines if "Reason:" in line][0].split("Reason:")[1].strip()
            
            selected_model = self.models.get(task_part)
            if selected_model is None:
                return None, None, f"알 수 없는 Task 선택됨: {task_part}"
                
            return selected_model, target_part, reason_part
            
        except Exception as e:
            return None, None, f"VLM이 지정된 포맷으로 대답하지 않았습니다. (에러: {e})"