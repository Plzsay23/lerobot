import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

class VLMClassifier(nn.Module):
    """VLM 위에 분류기(Linear Layer)를 붙인 커스텀 모델"""
    def __init__(self, vlm_model, num_classes=3):
        super().__init__()
        self.vlm = vlm_model
        
        # SmolVLM의 언어 모델 히든 사이즈를 가져옴 (일반적으로 2048 등)
        hidden_size = self.vlm.config.text_config.hidden_size
        
        # [핵심] 원핫 벡터를 뽑아낼 작은 어댑터 레이어 (분류기)
        # 나중에 파인튜닝할 때는 이 Linear 레이어(와 VLM의 LoRA)만 학습시킵니다.
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, inputs):
        # 1. VLM을 한 번만 통과시킴 (텍스트 생성 generate가 아님! 엄청 빠름)
        outputs = self.vlm(**inputs, output_hidden_states=True)
        
        # 2. 마지막 레이어의 마지막 토큰(가장 많은 정보를 담은 토큰)의 특징을 추출
        last_hidden_state = outputs.hidden_states[-1] # [Batch, Seq_Len, Hidden_Size]
        last_token_feature = last_hidden_state[:, -1, :] # [Batch, Hidden_Size]
        
        # 3. 분류기 통과시켜서 클래스 개수만큼의 확률값(Logits) 출력
        logits = self.classifier(last_token_feature) # [Batch, Num_Classes]
        return logits


class ManagerAgent:
    def __init__(self, available_models, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        self.models_list = list(available_models.keys()) # ['pick_place', 'cleaning']
        self.available_models = available_models
        
        print(f"🔄 [System] VLM 분류기(원핫 벡터 출력용) 로드 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        base_vlm = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to(self.device)
        base_vlm.eval()
        
        # 등록된 태스크 개수 + 1 (할 일 없음/None) 만큼의 클래스 지정
        self.num_classes = len(self.models_list) + 1 
        
        # 커스텀 분류기 모델 장착
        self.custom_vlm = VLMClassifier(base_vlm, num_classes=self.num_classes).to(self.device)
        
        # 현재는 파인튜닝이 안 된 "랜덤 상태"이므로 랜덤한 예측이 나옴
        self.custom_vlm.eval() 
        
        print("✅ [System] VLM 에이전트 준비 완료! (분류기 모드)")

    def observe_and_think(self, image_array, user_goal):
        import time
        start_time = time.time()
        print("🧠 [VLM 추론 중] 포워드 패스 실행 (매우 빠름)...")
        
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        image_pil = Image.fromarray(image_array)

        # 텍스트 생성이 아니므로 프롬프트를 아주 단순하게 줍니다.
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

        # [핵심] Generate()를 쓰지 않고 Forward() 한 번으로 끝냅니다.
        with torch.no_grad():
            logits = self.custom_vlm(inputs) # 출력: 예) [[-0.2, 1.5, 0.3]]
            
            # 가장 높은 값을 가진 인덱스 찾기
            predicted_class_idx = torch.argmax(logits, dim=-1).item()
            
            # 원핫 인코딩 벡터 생성 (테스트 출력용)
            one_hot = torch.zeros(self.num_classes, dtype=torch.int)
            one_hot[predicted_class_idx] = 1

        print(f"⏱️  [추론 시간] {time.time() - start_time:.2f}초")
        print(f"🤖 [VLM 출력 벡터] {one_hot.tolist()} (Logits: {logits[0].tolist()})")

        # 인덱스 해석
        if predicted_class_idx < len(self.models_list):
            task_name = self.models_list[predicted_class_idx]
            selected_model = self.available_models[task_name]
            reason = f"VLM이 클래스 {predicted_class_idx}번({task_name})을 선택했습니다 (현재는 랜덤)"
            # Target은 임시로 더미값 지정 (나중에 Object Detection 등과 연계 가능)
            target = "dummy_target" 
            return selected_model, target, reason
        else:
            return None, None, "VLM이 '작업 없음(None)' 클래스를 선택했습니다."