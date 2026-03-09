# manager_agent.py
import json
import time
from pathlib import Path

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
    def __init__(self, num_classes=None, model_id="HuggingFaceTB/SmolVLM-Instruct"):
        self.base_dir = Path(__file__).resolve().parent
        self.weight_path = self.base_dir / 'vlm_adapter_weights.pth'
        self.meta_path = self.base_dir / 'vlm_adapter_meta.json'
        self.meta = self._load_metadata()

        meta_num_classes = self.meta.get('num_classes')
        if num_classes is None:
            if meta_num_classes is None:
                raise ValueError('num_classes is required when metadata is missing')
            num_classes = meta_num_classes
        elif meta_num_classes is not None and meta_num_classes != num_classes:
            print(f"⚠️ [VLM] num_classes mismatch | arg={num_classes} meta={meta_num_classes}; using metadata")
            num_classes = meta_num_classes

        print(f"🔄 [VLM] 범용 VLM 분류기 로드 중... (총 {num_classes}개 클래스 인식)")
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = float(self.meta.get('confidence_threshold', 0.55))
        self.margin_threshold = float(self.meta.get('margin_threshold', 0.10))
        self.enable_abstain = bool(self.meta.get('enable_abstain', False))
        self.abstain_label = self.meta.get('abstain_label', 'none')
        self.labels = self.meta.get('labels', [str(i) for i in range(self.num_classes)])
        self.model_id = self.meta.get('model_id', model_id)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        base_vlm = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation='sdpa',
        ).to(self.device)
        base_vlm.eval()

        self.custom_vlm = VLMClassifier(base_vlm, num_classes=self.num_classes).to(self.device)
        self._load_classifier_weights()
        self.custom_vlm.eval()
        print('✅ [VLM] 로드 완료!')

    def _load_metadata(self):
        if not self.meta_path.exists():
            print(f"⚠️ [VLM] metadata file not found: {self.meta_path}")
            return {}
        try:
            return json.loads(self.meta_path.read_text())
        except Exception as e:
            print(f"⚠️ [VLM] failed to load metadata: {e}")
            return {}

    def _load_classifier_weights(self):
        if not self.weight_path.exists():
            print(f"⚠️ [VLM] fine-tuned classifier weights not found: {self.weight_path}")
            return

        state_dict = torch.load(self.weight_path, map_location=self.device)
        missing, unexpected = self.custom_vlm.classifier.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"⚠️ [VLM] classifier load mismatch | missing={missing} unexpected={unexpected}")
        else:
            print(f"✅ [VLM] classifier weights loaded from {self.weight_path}")

    def _prepare_image(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.detach().cpu().numpy()
        else:
            img_np = image_tensor

        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = img_np.transpose(1, 2, 0)
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np[..., 0]

        if img_np.dtype != 'uint8':
            max_val = float(img_np.max()) if img_np.size else 0.0
            if max_val <= 1.5:
                img_np = (img_np * 255).clip(0, 255).astype('uint8')
            else:
                img_np = img_np.clip(0, 255).astype('uint8')

        return Image.fromarray(img_np)

    def predict(self, image_tensor, instruction_text):
        print(f"🧠 [VLM] '{instruction_text}' 명령 분석 중...")
        image_pil = self._prepare_image(image_tensor)

        system_prompt = 'Classify the task into a specific index based on the image and instruction.'
        user_prompt = f'Instruction: {instruction_text}'
        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': system_prompt},
                {'type': 'image'},
                {'type': 'text', 'text': user_prompt},
            ]}
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image_pil], return_tensors='pt').to(self.device)

        start_time = time.time()
        with torch.no_grad():
            logits = self.custom_vlm(inputs)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
            top2 = torch.topk(probs, k=min(2, probs.numel()))

        top1_idx = int(top2.indices[0].item())
        top1_prob = float(top2.values[0].item())
        top2_prob = float(top2.values[1].item()) if top2.values.numel() > 1 else 0.0
        margin = top1_prob - top2_prob
        label = self.labels[top1_idx] if top1_idx < len(self.labels) else str(top1_idx)

        abstained = False
        abstain_reason = None
        if self.enable_abstain and label == self.abstain_label:
            abstained = True
            abstain_reason = 'predicted_abstain_label'
        elif top1_prob < self.confidence_threshold:
            abstained = True
            abstain_reason = 'low_confidence'
        elif margin < self.margin_threshold:
            abstained = True
            abstain_reason = 'low_margin'

        print(
            f"⏱️ [VLM] 판단 완료 ({time.time() - start_time:.2f}초) -> "
            f"idx={top1_idx} label={label} prob={top1_prob:.3f} margin={margin:.3f} abstained={abstained}"
        )

        return {
            'index': top1_idx,
            'label': label,
            'probability': top1_prob,
            'margin': margin,
            'abstained': abstained,
            'abstain_reason': abstain_reason,
            'probabilities': probs.tolist(),
            'labels': self.labels,
        }

    def predict_action_index(self, image_tensor, instruction_text):
        result = self.predict(image_tensor, instruction_text)
        return result['index']
