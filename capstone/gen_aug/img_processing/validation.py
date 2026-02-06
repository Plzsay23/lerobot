import pandas as pd
from datasets import load_dataset
import numpy as np

def verify_consistency(original_id, final_parquet_path):
    print("1. 두 데이터셋을 불러오는 중...")
    # 원본 데이터셋 로드
    ds_orig = load_dataset(original_id, split="train")
    # 사용자가 만든 최종 Parquet 로드
    ds_final = load_dataset("parquet", data_files=final_parquet_path, split="train")

    # 데이터 개수 확인
    print(f"원본 데이터 수: {len(ds_orig)}")
    print(f"최종 데이터 수: {len(ds_final)}")

    # 2. 첫 5행의 'action' 값 직접 비교
    print("\\n[상단 5행 Action 값 비교]")
    for i in range(5):
        orig_action = ds_orig[i]['action']
        final_action = ds_final[i]['action']

        # 리스트나 배열 형태일 것이므로 numpy로 비교
        is_same = np.allclose(orig_action, final_action)
        print(f"프레임 {i}: 원본={orig_action} | 최종={final_action} | 일치여부: {is_same}")

    # 3. 전체 데이터셋에 대한 수치적 오차 합계 계산
    print("\\n3. 전체 데이터셋 수치 검증 중...")
    orig_actions = np.array(ds_orig['action'])
    final_actions = np.array(ds_final['action'])

    # 두 배열의 차이 계산 (0에 가까울수록 완벽히 일치)
    difference = np.sum(np.abs(orig_actions - final_actions))

    if difference == 0:
        print("✅ 성공: 모든 좌표값이 원본과 100% 일치합니다!")
    else:
        print(f"⚠️ 경고: 수치적 차이가 발견되었습니다. (누적 오차: {difference})")
    
if __name__ == "main":
    verify_consistency("lerobot/pusht_image", "pusht_genaug_final.parquet")
