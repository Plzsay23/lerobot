# manager_agent.py
import time
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class VLMClassifier(nn.Module):
    """VLM 본체에 동적 크기의 분류기(Linear)를 붙인 어댑터 모델"""
    def __init__(self, vlm_model, num_classes):
        super().__init__()
        self.vlm = vlm_model
        hidden_size = self.vlm.config.text_config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes).to(dtype=torch.bfloat16)
        
    def forward(self, inputs):
        outputs = self.vlm(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] 
        last_token_feature = last_hidden_state[:, -1, :] 
        logits = self.classifier(last_token_feature) 
        return logits

class ManagerAgent:
    def __init__(self, num_classes, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        """
        num_classes: ACT 모델 개수 + 1 (마지막은 '해당 작업 없음/대기' 클래스)
        """
        print(f"🔄 [VLM] 범용 VLM 분류기 로드 중... (총 {num_classes}개 클래스 인식)")
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        base_vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(self.device)
        base_vlm.eval()
        
        self.custom_vlm = VLMClassifier(base_vlm, num_classes=self.num_classes).to(self.device)
        self.custom_vlm.eval() 
        print("✅ [VLM] 로드 완료!")

    def predict_action_index(self, image_tensor, instruction_text):
        """
        로봇 카메라 텐서와 텍스트를 받아 모델 인덱스(정수)를 반환
        """
        print(f"🧠 [VLM] '{instruction_text}' 명령 분석 중...")
        
        # 로봇 파이프라인에서 온 텐서 [C, H, W] 를 PIL 이미지로 변환
        if isinstance(image_tensor, torch.Tensor):
            # 보통 정규화되어 있을 수 있으므로 처리 (0~1 범위 가정)
            img_np = image_tensor.cpu().numpy()
            img_np = (img_np.transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
            image_pil = Image.fromarray(img_np)
        else:
            image_pil = Image.fromarray(image_tensor)

        system_prompt = "Classify the task into a specific index based on the image and instruction."
        user_prompt = f"Instruction: {instruction_text}"

        messages = [
            {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image"}, {"type": "text", "text": user_prompt}]}
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image_pil], return_tensors="pt").to(self.device)

        start_time = time.time()
        with torch.no_grad():
            logits = self.custom_vlm(inputs)
            predicted_class_idx = torch.argmax(logits, dim=-1).item()

        print(f"⏱️ [VLM] 판단 완료 ({time.time() - start_time:.2f}초) -> 클래스 Index: [{predicted_class_idx}]")
        return predicted_class_idx