from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "google/gemma-3-1b-it"

def load_model(model_id=MODEL_ID):
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=os.environ.get("HUGGING_FACE_TOKEN"),
        trust_remote_code=True,
        use_fast=False,
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=os.environ.get("HUGGING_FACE_TOKEN"),
        trust_remote_code=True,
        device_map="auto"
    )
    
    # CPU에서 실행 시 명시적으로 모델을 디바이스로 이동
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    return model, tokenizer, device

def generate_summary(treatment_text, model=None, tokenizer=None, device=None, max_new_tokens=2000):
    # 모델과 토크나이저를 로드하지 않았다면 로드
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_model()
    
    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": "당신은 마취통증의학과 전문의입니다. 아래 상담 내용을 요약해 주세요. 형식은 아래와 같습니다:\n\n- 증상:\n- 진단:\n- 치료:\n- 주의사항:\n- 환자 반응:\n\n반드시 형식을 지킨 요약 내용만 출력해 주세요."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": treatment_text}]
        }
    ]]

    # 입력 생성
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True
    )
    
    # 추론 타입 설정 (GPU인 경우 bfloat16, CPU인 경우 float32)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    inputs = inputs.to(model.device)

    # 생성
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 결과 디코딩
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 모델 응답 추출
    if "assistant" in result:
        result = result.split("assistant")[-1].strip()
    
    return result

# 미리 모델 로드 (앱 시작 시 한 번만 실행됨)
# model, tokenizer, device = load_model() 

model = None
tokenizer = None
device = None

def get_model():
    global model, tokenizer, device
    if model is None:
        model, tokenizer, device = load_model()
    return model, tokenizer, device