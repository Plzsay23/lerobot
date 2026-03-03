# manager_agent.py
import torch
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class ManagerAgent:
    def __init__(self, available_models, pretrained_path="lerobot/smolvla_base"):
        self.models = available_models
        print(f"🔄 [System] SmolVLA 정책 로드 중... ({pretrained_path})")
        
        # 1. LeRobot의 SmolVLA Policy 로드 (모터 제어용 포함 전체 모델)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        self.policy.to(device)
        self.policy.eval()
        
        # 2. VLM 에이전트로 쓰기 위해 순수 VLM과 Processor만 추출
        self.vlm = self.policy.model.vlm_with_expert.vlm
        self.processor = self.policy.model.vlm_with_expert.processor
        self.device = device
        print("✅ [System] VLM 에이전트 준비 완료!")

    def observe_and_think(self, image_array, user_goal):
        """
        실제 카메라 이미지(NumPy/Tensor)와 텍스트 명령을 받아 VLM으로 추론
        """
        print("🧠 [VLM 추론 중] 카메라 이미지와 명령어를 분석합니다...")
        
        # 이미지를 PIL 형식으로 변환 (Processor 입력용)
        # lerobot 카메라는 보통 RGB 형태의 numpy array (H, W, 3)을 반환합니다.
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        image_pil = Image.fromarray(image_array)

        # VLM에게 내릴 프롬프트 작성 (역할, 룰, 입력값 지정)
        system_prompt = (
            "You are a robotic task manager. Based on the user's instruction and the image, "
            "decide which task to execute. Available tasks are: 'pick_place' and 'cleaning'. "
            "Output the result exactly in this format: 'Task: [task_name], Target: [object_name], Reason: [your reason]'"
        )
        user_prompt = f"Instruction: {user_goal}"

        # SmolVLM Instruct 모델의 Chat Template 적용
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
        
        # Processor를 통해 입력 텐서 생성
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image_pil], return_tensors="pt").to(self.device)

        # VLM 추론 실행 (텍스트 생성)
        with torch.no_grad():
            generated_ids = self.vlm.generate(**inputs, max_new_tokens=50)
        
        # 결과 디코딩 (입력 프롬프트 부분은 제외하고 생성된 텍스트만 추출)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        print(f"🤖 [VLM Raw Output] {output_text}")

        # 출력 텍스트 파싱 (Task, Target, Reason 추출)
        try:
            task_part = output_text.split("Task:")[1].split(",")[0].strip()
            target_part = output_text.split("Target:")[1].split(",")[0].strip()
            reason_part = output_text.split("Reason:")[1].strip()
            
            selected_model = self.models.get(task_part)
            if selected_model is None:
                return None, None, f"알 수 없는 Task 선택됨: {task_part}"
                
            return selected_model, target_part, reason_part
            
        except IndexError:
            return None, None, f"VLM이 지정된 포맷으로 대답하지 않았습니다. (출력: {output_text})"