import os
import pandas as pd
import numpy as np


# ==============================
# 경로 설정
# ==============================

input_csv = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\PPG_myocardial-infraction\PPG_Dataset.csv"
save_root = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\PPG_myocardial-infraction\Data"

sampling_rate = 125
test_size_per_class = 200


# ==============================
# 데이터 로드
# ==============================

df = pd.read_csv(input_csv)

print("Dataset shape:", df.shape)

signal_columns = df.columns[:-1]
label_column = df.columns[-1]


# ==============================
# label별 index 분리
# ==============================

normal_idx = df[df[label_column] == "Normal"].index
mi_idx = df[df[label_column] == "MI"].index


# ==============================
# test 샘플 선택
# ==============================

np.random.seed(42)

test_normal = np.random.choice(normal_idx, test_size_per_class, replace=False)
test_mi = np.random.choice(mi_idx, test_size_per_class, replace=False)

test_indices = set(test_normal.tolist() + test_mi.tolist())


# ==============================
# 저장 폴더 생성
# ==============================

for split in ["train", "test"]:
    for label in ["Normal", "MI"]:
        os.makedirs(os.path.join(save_root, split, label), exist_ok=True)


# ==============================
# 데이터 저장
# ==============================

for idx, row in df.iterrows():

    signal = row[signal_columns].values.astype(float)
    label = row[label_column]

    # train / test 결정
    if idx in test_indices:
        split = "test"
    else:
        split = "train"

    save_dir = os.path.join(save_root, split, label)

    subject_id = f"subject_{idx:05d}.csv"

    t = np.arange(len(signal)) / sampling_rate

    save_df = pd.DataFrame({
        "time": t,
        "ppg": signal
    })

    save_path = os.path.join(save_dir, subject_id)

    save_df.to_csv(save_path, index=False)


print("저장 완료")