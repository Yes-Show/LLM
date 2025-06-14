import os
from pyngrok import ngrok, conf
import uvicorn
from app.main import app
from dotenv import load_dotenv
load_dotenv()

# 토큰 설정
auth_token = os.getenv("NGROK_AUTH_TOKEN")
if auth_token:
    conf.get_default().auth_token = auth_token
    print("✅ Ngrok 인증 완료")
else:
    print("⚠️ NGROK_AUTH_TOKEN 환경변수가 설정되지 않았습니다.")

# 포트와 커스텀 서브도메인 설정
port = int(os.getenv("PORT", "8000"))
domain = os.getenv("NGROK_STATIC_DOMAIN", "ladybird-needed-lately.ngrok-free.app")

# Ngrok 터널 (고정 도메인 사용)
public_url = ngrok.connect(
    addr=port,
    hostname=domain,
    bind_tls=True  # HTTPS
)
print(f"🌐 Ngrok 고정 도메인 활성화: {public_url}")

# FastAPI 실행
uvicorn.run(app, host="0.0.0.0", port=port)