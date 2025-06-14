# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.10-slim
WORKDIR /app

# 기본 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 애플리케이션 코드 복사
COPY app app
# Python 기본 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app

EXPOSE 8000

# Ngrok과 FastAPI 실행
CMD ["python3", "app/run.py"]