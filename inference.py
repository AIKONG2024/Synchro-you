import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from typing import List

# FastAPI 앱 초기화
app = FastAPI()

# S3에서 모델 파일 다운로드
def download_model_from_s3(bucket_name, model_key, model_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_key, model_path)

# 모델 불러오기 함수
# def load_scripted_model(model_path):
#     model = torch.jit.load(model_path)
#     model.eval()
#     return model

# cpu test
def load_scripted_model(model_path):
    model = torch.jit.load(model_path, map_location="cpu")  # CPU에서 강제로 로드
    model.eval()
    return model

# S3 버킷과 모델 정보 설정
bucket_name = 'synchro-you-bucket'
model_key = 'synchro-you-lstm-model.pt'  # TorchScript 모델 경로
model_path = 'synchro-you-lstm-model.pt'

# S3에서 모델 다운로드 및 로드
download_model_from_s3(bucket_name, model_key, model_path)
model = load_scripted_model(model_path)

# 요청 데이터 형식 정의
class PredictionRequest(BaseModel):
    input_data: List[List[float]]

# health 엔드포인트
@app.get("/health")
async def health_check():
    return {"status":"ok"}

# 추론 엔드포인트 설정
@app.post("/predict")
async def predict(request: PredictionRequest):
    data = request.input_data
    if not data:
        raise HTTPException(status_code=400, detail="No input data provided")

    # 입력 데이터를 텐서로 변환 및 GPU 할당
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Batch dimension 추가
    input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 추론 수행
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.cpu().numpy().tolist()

    return {"prediction": prediction}
