import os
import sys
import random
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
async_inference_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(os.path.dirname(async_inference_dir))

sys.path.insert(0, async_inference_dir)
sys.path.insert(0, src_dir)

import av
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from PIL import Image
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoProcessor

from manager_agent import VLMClassifier


def resolve_dataset_root(repo_id: str) -> Path:
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return Path(local_dir)


def list_camera_videos(dataset_root: Path, camera_key: str) -> list[Path]:
    video_dir = dataset_root / "videos" / camera_key
    if not video_dir.exists():
        return []
    return sorted(video_dir.rglob("*.mp4"))


def decode_random_frame(video_path: Path) -> Image.Image | None:
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total_frames = stream.frames
            if total_frames and total_frames > 0:
                target_index = random.randint(0, total_frames - 1)
            else:
                target_index = None

            decoded = []
            for i, frame in enumerate(container.decode(stream)):
                if target_index is not None:
                    if i == target_index:
                        return frame.to_image()
                else:
                    decoded.append(frame)

            if decoded:
                return random.choice(decoded).to_image()
    except Exception as e:
        print(f"⚠️ 비디오 프레임 추출 실패: {video_path} | {e}")
    return None


def sample_images_from_dataset(repo_id: str, camera_key: str, num_samples: int) -> list[Image.Image]:
    dataset_root = resolve_dataset_root(repo_id)
    videos = list_camera_videos(dataset_root, camera_key)
    if not videos:
        raise FileNotFoundError(f"No videos found for camera '{camera_key}' under {dataset_root}")

    sampled_images = []
    while len(sampled_images) < num_samples:
        video_path = random.choice(videos)
        image = decode_random_frame(video_path)
        if image is not None:
            sampled_images.append(image.convert("RGB"))
    return sampled_images


def main():
    print("🚀 Hugging Face 데이터셋을 활용한 VLM 파인튜닝 시작...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_configs = [
        {"repo_id": "plzsay/green", "text": "pick up the green block", "label": 0},
        {"repo_id": "plzsay/black", "text": "pick up the black block", "label": 1},
        {"repo_id": "plzsay/blue", "text": "pick up the blue block", "label": 2},
    ]

    camera_key = "observation.images.top"
    num_samples_per_task = 30
    train_data = []

    for config in dataset_configs:
        repo_id = config["repo_id"]
        print(f"\n📥 [{repo_id}] 데이터셋 로드 중...")
        images = sample_images_from_dataset(repo_id, camera_key, num_samples_per_task)
        for image in images:
            train_data.append({
                "image": image,
                "text": config["text"],
                "label": config["label"],
            })
        print(f"✅ [{repo_id}] 에서 {len(images)}장의 프레임을 추출했습니다.")

    random.shuffle(train_data)
    print(f"\n📊 총 학습 데이터 수: {len(train_data)}장")

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

    epochs = 5
    print("\n🔥 학습을 시작합니다...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for batch in train_data:
            system_prompt = "Classify the task into a specific index based on the image and instruction."
            user_prompt = f"Instruction: {batch['text']}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
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

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_data):.4f} | Acc: {correct/len(train_data)*100:.1f}%"
        )

    save_path = os.path.join(async_inference_dir, "vlm_adapter_weights.pth")
    torch.save(model.classifier.state_dict(), save_path)
    print(f"\n🎉 파인튜닝 완료! 가중치가 '{save_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
