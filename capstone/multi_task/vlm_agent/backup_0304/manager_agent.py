import time
import torch
import torch.nn as nn
from PIL import Image
# 경고 메시지 방지를 위해 AutoModelForImageTextToText로 변경
from transformers import AutoProcessor, AutoModelForImageTextToText

class VLMClassifier(nn.Module):
    """VLM 위에 분류기(Linear Layer)를 붙인 커스텀 모델"""
    def __init__(self, vlm_model, num_classes=3):
        super().__init__()
        self.vlm = vlm_model
        
        # SmolVLM의 언어 모델 히든 사이즈 가져오기
        hidden_size = self.vlm.config.text_config.hidden_size
        
        # [완벽 해결] Linear 레이어를 만든 직후 명시적으로 bfloat16으로 강제 캐스팅합니다.
        self.classifier = nn.Linear(hidden_size, num_classes).to(dtype=torch.bfloat16)
        
    def forward(self, inputs):
        # 1. VLM을 한 번만 통과시킴 (텍스트 생성 아님!)
        outputs = self.vlm(**inputs, output_hidden_states=True)
        
        # 2. 마지막 레이어의 마지막 토큰(가장 많은 정보를 담은 토큰) 특징 추출
        last_hidden_state = outputs.hidden_states[-1] 
        last_token_feature = last_hidden_state[:, -1, :] 
        
        # 3. 분류기 통과
        logits = self.classifier(last_token_feature) 
        return logits

class ManagerAgent:
    def __init__(self, available_models, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        self.models_list = list(available_models.keys())
        self.available_models = available_models
        
        print(f"🔄 [System] VLM 분류기(원핫 벡터 출력용) 로드 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # [수정됨] 최신 클래스로 변경하고 dtype 파라미터 사용
        base_vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(self.device)
        base_vlm.eval()
        
        self.num_classes = len(self.models_list) + 1 
        
        # 커스텀 분류기 모델 장착
        self.custom_vlm = VLMClassifier(base_vlm, num_classes=self.num_classes).to(self.device)
        self.custom_vlm.eval() 
        
        print("✅ [System] VLM 에이전트 준비 완료! (분류기 모드)")

    def observe_and_think(self, image_array, user_goal):
        start_time = time.time()
        print("🧠 [VLM 추론 중] 포워드 패스 실행 (단일 연산)...")
        
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        image_pil = Image.fromarray(image_array)

        system_prompt = "Classify the task based on the image and instruction."
        user_prompt = f"Instruction: {user_goal}"

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

        # Generate가 아닌 Forward 호출
        with torch.no_grad():
            logits = self.custom_vlm(inputs)
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            
            one_hot = torch.zeros(self.num_classes, dtype=torch.int)
            one_hot[predicted_class_idx] = 1

        print(f"⏱️  [추론 시간] {time.time() - start_time:.2f}초")
        print(f"🤖 [VLM 출력 벡터] {one_hot.tolist()} (Logits: {logits[0].tolist()})")

        if predicted_class_idx < len(self.models_list):
            task_name = self.models_list[predicted_class_idx]
            selected_model = self.available_models[task_name]
            reason = f"VLM이 클래스 {predicted_class_idx}번({task_name})을 선택했습니다 (현재는 랜덤 가중치)"
            target = "dummy_target" 
            return selected_model, target, reason
        else:
            return None, None, "VLM이 '작업 없음(None)' 클래스를 선택했습니다."