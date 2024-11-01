# # 기본 이미지 설정 - PyTorch와 Flask가 사전 설치된 이미지 사용
# FROM pytorch/torchserve:latest

# # 작업 디렉토리 설정
# WORKDIR /app

# # 라이브러리 설치
# RUN pip install fastapi uvicorn boto3

# # 애플리케이션 파일 복사
# COPY ./model/synchro-you-lstm-model.pt /app/synchro-you-lstm-model.pt
# COPY inference.py /app/inference.py

# # 포트 노출
# EXPOSE 8000

# # 서버 실행 명령어 설정
# CMD ["python3", "inference.py"]

##local 확인용
# 기본 이미지 설정
# Use the base image
# FROM pytorch/torchserve:latest

# # Switch to root user
# USER root

# # Set the working directory
# WORKDIR /app

# # Install libraries
# RUN pip install fastapi uvicorn boto3 pandas

# # Copy application files
# COPY ./model/synchro-you-lstm-model.pt /app/synchro-you-lstm-model.pt
# COPY inference.py /app/inference.py

# # Copy SSL certificate files
# COPY ./keys/private.key /app/private.key
# COPY ./keys/certificate.crt /app/certificate.crt

# # Change permissions of the certificate files
# RUN chmod 644 /app/certificate.crt /app/private.key

# # Switch back to the original user (if necessary)
# # USER your_original_user

# # Expose port 443
# EXPOSE 80

# # Run the FastAPI app with Uvicorn over HTTPS
# CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "/app/private.key", "--ssl-certfile", "/app/certificate.crt"]


#HTTP
FROM pytorch/torchserve:latest

# Switch to root user
USER root

# Set the working directory
WORKDIR /app

# Install libraries
RUN pip install fastapi uvicorn boto3 pandas

# Copy application files
COPY ./model/traced_model_script_cpu.pt /app/traced_model_script_cpu.pt
COPY ./data/anchor_embedding.npy /app/anchor_embedding.npy
COPY inference.py /app/inference.py

# Expose port 8000 for HTTP
EXPOSE 8000

# Run the FastAPI app with Uvicorn over HTTP
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]
