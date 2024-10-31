import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from typing import List

app = FastAPI()

# S3에서 모델 가져오기
# def download_model_from_s3(bucket_name, model_key, model_path):
#     s3 = boto3.client('s3')
#     model = s3.download_file(bucket_name, model_key, model_path)
#     return model

# Docker Image에서 모델 가져오기
def load_scripted_model(model_path):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model

# bucket_name = 'synchro-you-bucket'
# model_key = 'synchro-you-lstm-model.pt'
model_path = 'synchro-you-lstm-model.pt'
# model = download_model_from_s3(bucket_name, model_key, model_path)

model = load_scripted_model(model_path)

# 두 개의 pose 데이터를 받기 위해 PredictionRequest 클래스
class PredictionRequest(BaseModel):
    user_pose_data: List[List[float]]
    target_pose_data: List[List[float]]
    
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

# 각도 계산
def calculate_angles(matrix1: pd.DataFrame, matrix2: pd.DataFrame):
    dot_product = np.einsum('ij,ij->i', matrix1, matrix2)
    norm1 = np.linalg.norm(matrix1, axis=1)
    norm2 = np.linalg.norm(matrix2, axis=1)
    cos_theta = dot_product / (norm1 * norm2)
    angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angles

# 각도 df 처리
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

# pretrained LSTM 모델을 이용한 embedding
def embedding(angle_df: pd.DataFrame):
    input_tensor = torch.tensor(angle_df.iloc[:44].values).unsqueeze(0).to(torch.float32).to('cpu')
    return model(input_tensor).detach().cpu().numpy()

# 유사도 계산
def calculate_similarity(vector1, vector2):
    norm1 = torch.nn.functional.normalize(torch.tensor(vector1), dim=1)
    norm2 = torch.nn.functional.normalize(torch.tensor(vector2), dim=1)
    distance = torch.nn.functional.pairwise_distance(norm1, norm2)
    return (distance ** 2).item()

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # user_pose_df 전처리 및 임베딩 생성
        user_pose_df = pd.DataFrame(request.user_pose_data)
        user_angles = make_df_angle(user_pose_df)
        user_emb = embedding(user_angles)

        # target_pose_data 전처리 및 임베딩 생성
        target_pose_data = pd.DataFrame(request.target_pose_data)
        target_angles = make_df_angle(target_pose_data)
        target_emb = embedding(target_angles)

        # 유사도 계산
        similarity = calculate_similarity(user_emb, target_emb)

        return {"similarity": similarity}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
