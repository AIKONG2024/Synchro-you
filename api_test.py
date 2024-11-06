import pandas as pd
import requests
from typing import List

# 30 프레임씩 데이터를 묶기
frame_size = 30
user_pose_data_batches = []

# 업로드된 파일을 로드
file_path = 'data/landmarks_3d_pos2_infer.csv'
data = pd.read_csv(file_path)

# 필요한 열만 선택 (3D 좌표만 추출)
pose_data = data.iloc[:, 1:]  # 첫 열은 filename이므로 제외

# 30 프레임씩 나누기
for i in range(0, len(pose_data), frame_size):
    batch = pose_data.iloc[i:i+frame_size].values.tolist()
    if len(batch) == frame_size:  # 30 프레임에 맞는 경우만 추가
        user_pose_data_batches.append(batch)

# 테스트할 첫 번째 batch 준비
user_pose_data_sample = user_pose_data_batches[0]
seq = 1  # 비교할 target embedding의 시점

# 요청을 보낼 URL 설정 (로컬 FastAPI 서버 가정)
url = "http://nipa-model-alb-804249540.ap-northeast-2.elb.amazonaws.com/predict"
# url = "http://localhost:8000/predict"

# 요청 payload 생성
payload = {
    "user_pose_data": user_pose_data_sample,
    "seq": seq
}

# 테스트 요청 보내기
try:
    response = requests.post(url, json=payload)
    response_data = response.json()
    print(response_data)
except requests.exceptions.RequestException as e:
    str(e)
