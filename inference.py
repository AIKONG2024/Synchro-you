import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 모델 로드
def load_scripted_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)  # 디바이스 지정
    model.to(device)
    model.eval()
    return model

model_path = 'traced_model_script_cpu.pt'
model = load_scripted_model(model_path)

target_npy_path = '/app/anchor_embedding.npy'

class PredictionRequest(BaseModel):
    user_pose_data: List[List[float]]
    seq: int

# 각도를 계산할 신체 부위 쌍과 좌표 인덱스 정의
angle = [['left_biceps', 'left_forearm'],
         ['right_biceps', 'right_forearm'],
         ['between_shoulders', 'left_body'],
         ['between_shoulders', 'right_body'],
         ['between_shoulders', 'rigth_neck'],
         ['between_shoulders', 'left_neck'],
         ['between_pelvis','left_thigh'],
         ['between_pelvis','right_thigh'],
         ['right_thigh','right_calf'],
         ['left_thigh','left_calf'],
         ['right_body','right_thigh'],
         ['left_body','left_thigh']
        ]
         
body_parts = {'left_biceps': [11, 13],
              'left_forearm': [13, 15],
              'right_biceps': [12, 14],
              'right_forearm': [14, 16],
              'between_shoulders': [11, 12],
              'left_body': [11, 23],
              'right_body': [12, 24],
              'between_pelvis': [23, 24],
              'left_thigh': [23, 25],
              'left_calf': [25, 27],
              'right_thigh': [24, 26],
              'right_calf': [26, 28],
              'left_neck': [9, 11],
              'rigth_neck': [10, 12]}

# 각도 계산 함수
def calculate_angles(matrix1: pd.DataFrame, matrix2: pd.DataFrame):
    dot_product = np.einsum('ij,ij->i', matrix1, matrix2)
    norm1 = np.linalg.norm(matrix1, axis=1)
    norm2 = np.linalg.norm(matrix2, axis=1)
    cos_theta = dot_product / (norm1 * norm2)
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angles

# 각도 데이터프레임 생성
def make_df_angle(pose_df: pd.DataFrame) -> pd.DataFrame:
    df_angle = pd.DataFrame()
    for body_parts1, body_parts2 in angle:
        vec_mat1 = pose_df.iloc[:, body_parts[body_parts1][0]*3+1:body_parts[body_parts1][0]*3+4].values \
                   - pose_df.iloc[:, body_parts[body_parts1][1]*3+1:body_parts[body_parts1][1]*3+4].values
        vec_mat2 = pose_df.iloc[:, body_parts[body_parts2][0]*3+1:body_parts[body_parts2][0]*3+4].values \
                   - pose_df.iloc[:, body_parts[body_parts2][1]*3+1:body_parts[body_parts2][1]*3+4].values
        df_angle[f'{body_parts1}_{body_parts2}'] = calculate_angles(vec_mat1, vec_mat2)
    df_angle = df_angle.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df_angle

# 임베딩 함수
def embedding(angle_df: pd.DataFrame):
    input_tensor = torch.tensor(angle_df.values).unsqueeze(0).to(torch.float32).to('cpu')
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy()

# 유사도 계산 함수
def calculate_similarity(vector1, vector2):
    vector1 = torch.tensor(vector1, dtype=torch.float32)
    vector2 = torch.tensor(vector2, dtype=torch.float32)

    # 텐서가 1차원인 경우 배치 차원을 추가하여 2차원으로 변환
    if vector1.dim() == 1:
        vector1 = vector1.unsqueeze(0)  # shape: [1, N]
    if vector2.dim() == 1:
        vector2 = vector2.unsqueeze(0)  # shape: [1, N]

    norm1 = torch.nn.functional.normalize(vector1, dim=1)
    norm2 = torch.nn.functional.normalize(vector2, dim=1)
    distance = torch.nn.functional.pairwise_distance(norm1, norm2, p=2)
    similarity = 1 - (distance.item() / 2)  # 유사도를 0~1 사이로 정규화
    return similarity

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # `user_pose_data`를 DataFrame으로 변환 후 각도 계산
        user_pose_df = pd.DataFrame(request.user_pose_data)
        user_angles = make_df_angle(user_pose_df)
        
        # `user_pose_data` 임베딩
        user_emb = embedding(user_angles)

        # `target embedding` 파일 로드 및 특정 시점 추출
        target_emb = np.load(target_npy_path)
        target_emb_seq = target_emb[request.seq]  # n 시점에서의 target 임베딩 추출

        # 유사도 계산
        similarity = calculate_similarity(user_emb, target_emb_seq)

        return {"similarity": similarity}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "connected"}
