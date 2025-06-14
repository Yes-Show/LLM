from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from app.model import generate_summary, model, tokenizer, device
from app.model import get_model, generate_summary
import logging
import time
import torch
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma 3 의료 요약 API",
    description="마취통증의학과 상담 내용을 요약하는 Gemma 3 모델 API",
    version="1.0.0"
)

class InputText(BaseModel):
    text: str
    max_new_tokens: int = 256

class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_id": "google/gemma-3-1b-it",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict")
async def predict(data: InputText):
    model, tokenizer, device = get_model()
    start_time = time.time()
    logger.info(f"Received request with {len(data.text)} characters")
    
    try:
        result = generate_summary(
            data.text, 
            model=model, 
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=data.max_new_tokens
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        return {
            "result": result,
            "processing_time_seconds": round(processing_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return JSONResponse(
        content={
            "message": "Gemma 3 의료 요약 API에 오신 것을 환영합니다",
            "endpoints": {
                "health": "/health",
                "predict": "/predict"
            }
        },
        media_type="application/json; charset=utf-8"
    )