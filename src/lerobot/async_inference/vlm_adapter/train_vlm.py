import os
import sys

# ==============================================================================
# [핵심 수정] 파이썬 경로(Path) 자동 주입기
# 현재 파일(train_vlm.py)의 위치를 기준으로 상위 폴더들을 파이썬 시스템에 강제로 인식시킵니다.
# 이렇게 하면 폴더 깊이에 상관없이 어디서 실행하든 에러가 발생하지 않습니다.
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))      # vlm_adapter 폴더
async_inference_dir = os.path.dirname(current_dir)            # async_inference 폴더
src_dir = os.path.dirname(os.path.dirname(async_inference_dir)) # src 폴더

sys.path.insert(0, async_inference_dir) # manager_agent.py를 찾기 위함
sys.path.insert(0, src_dir)             # lerobot 코어 모듈을 찾기 위함

import torch
import torch.nn as nn
from torch.optim import AdamW
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

# 상위 폴더에 있는 VLM 분류기 무사히 가져오기
from manager_agent import VLMClassifier

# ==============================================================================
# [핵심 수정] LeRobot 버전에 따른 데이터셋 로더 안전망 (Try-Except)
# ==============================================================================
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.dataset import LeRobotDataset


def get_pil_from_tensor(img_tensor):
    """LeRobot 데이터셋의 텐서 [C, H, W] (0.0~1.0)를 PIL 이미지로 변환"""
    img_np = img_tensor.cpu().numpy()
    img_np = (img_np.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

def main():
    print("🚀 Hugging Face 데이터셋을 활용한 VLM 파인튜닝 시작...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 학습할 Hugging Face 데이터셋 및 라벨 맵핑 정의
    dataset_configs = [
        {"repo_id": "plzsay/green", "text": "pick up the green block", "label": 0},
        {"repo_id": "plzsay/black", "text": "pick up the black block", "label": 1},
        {"repo_id": "plzsay/blue",  "text": "pick up the blue block",  "label": 2},
    ]
    
    camera_key = "observation.images.top" 
    num_samples_per_task = 30 # 추출할 프레임 수
    train_data = []

    # 2. LeRobotDataset으로 동영상에서 이미지 프레임 추출
    for config in dataset_configs:
        repo_id = config["repo_id"]
        print(f"\n📥 [{repo_id}] 데이터셋 로드 중...")
        
        dataset = LeRobotDataset(repo_id)
        total_frames = len(dataset)
        sample_indices = np.random.choice(total_frames, min(num_samples_per_task, total_frames), replace=False)
        
        for idx in sample_indices:
            frame_data = dataset[idx]
            
            if camera_key in frame_data:
                img_tensor = frame_data[camera_key]
                img_pil = get_pil_from_tensor(img_tensor)
                
                train_data.append({
                    "image": img_pil,
                    "text": config["text"],
                    "label": config["label"]
                })
            else:
                print(f"⚠️ 데이터셋에 '{camera_key}'가 없습니다. 사용 가능한 키: {list(frame_data.keys())}")
                break
                
        print(f"✅ [{repo_id}] 에서 {len(sample_indices)}장의 프레임을 추출했습니다.")

    np.random.shuffle(train_data)
    print(f"\n📊 총 학습 데이터 수: {len(train_data)}장")

    # 3. VLM 로드 및 얼리기 (Freeze)
    num_classes = len(dataset_configs) + 1 
    
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    base_vlm = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    
    for param in base_vlm.parameters():
        param.requires_grad = False

    model = VLMClassifier(base_vlm, num_classes=num_classes).to(device)
    model.train()
    
    optimizer = AdamW(model.classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4. 학습 루프
    epochs = 5 
    print("\n🔥 학습을 시작합니다...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for batch in train_data:
            system_prompt = "Classify the task into a specific index based on the image and instruction."
            user_prompt = f"Instruction: {batch['text']}"
            
            messages = [
                {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image"}, {"type": "text", "text": user_prompt}]}
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[batch["image"]], return_tensors="pt").to(device)
            
            target = torch.tensor([batch["label"]], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if torch.argmax(logits, dim=-1).item() == batch["label"]:
                correct += 1
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_data):.4f} | Acc: {correct/len(train_data)*100:.1f}%")

    # 5. 가중치 저장 (manager_agent.py가 읽을 수 있는 위치에 저장)
    save_path = os.path.join(async_inference_dir, "vlm_adapter_weights.pth")
    torch.save(model.classifier.state_dict(), save_path)
    print(f"\n🎉 파인튜닝 완료! 가중치가 '{save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    main()