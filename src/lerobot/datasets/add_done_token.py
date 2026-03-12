import os
import numpy as np
from datasets import load_dataset

# 녹화할 때 사용했던 HF_USER/TASK_NAME을 적어주세요.
DATASET_ID = "plzsay/black"

# 1.0(Done Token)이 추가되어 새로 저장될 폴더 경로
SAVE_PATH = "./dataset_with_done_token_black"
# ==========================================

def main():
    print(f"[{DATASET_ID}] 원본 데이터셋을 불러오는 중...")
    
    try:
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return

    print(" 에피소드별 마지막 프레임을 탐색 중...")
    episode_indices = dataset["episode_index"]
    is_last_frame = []
    
    total_frames = len(episode_indices)
    for i in range(total_frames):
        # 현재 프레임이 전체 데이터의 맨 마지막이거나, 
        # 다음 프레임에서 에피소드 번호가 바뀌면 -> 해당 에피소드의 마지막!
        if i == total_frames - 1 or episode_indices[i] != episode_indices[i+1]:
            is_last_frame.append(True)
        else:
            is_last_frame.append(False)

    print(f" 총 {total_frames}프레임 중 {sum(is_last_frame)}개의 에피소드 종료 지점을 찾았습니다.")

    # Action 차원 늘리기 (6D -> 7D)
    def process_action(example, idx):
        # 기존 action 배열
        action = np.array(example["action"], dtype=np.float32)
        
        # 마지막 프레임이면 1.0 (Done), 아니면 0.0
        done_signal = 1.0 if is_last_frame[idx] else 0.0
        
        # 배열 끝에 Done 신호를 추가하여 차원 확장
        new_action = np.append(action, done_signal)
        
        example["action"] = new_action
        return example

    print(" Action 차원 확장 (Done Token 1.0 추가) 진행 중...")
    updated_dataset = dataset.map(process_action, with_indices=True)

    print(" 새로운 데이터셋을 디스크에 저장하는 중...")
    updated_dataset.save_to_disk(SAVE_PATH)
    print(f"\n 처리가 끝났습니다! 확장된 데이터셋이 '{SAVE_PATH}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
