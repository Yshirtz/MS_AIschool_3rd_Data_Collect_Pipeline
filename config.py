import os
from dotenv import load_dotenv

# =========================================================
# 0. 환경 변수(.env) 로드
# =========================================================
load_dotenv()

# =========================================================
# 1. 기본 경로 및 모델 경로 설정
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
SAM_MODEL_NAME = os.path.join(MODELS_DIR, "sam2.1_b.pt")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "resnet50_triplet_final.pth")

# =========================================================
# 2. AI 모델 파라미터 및 모드 설정
# =========================================================
CONF_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.3
USE_DUMMY_TEXT_EMBEDDING = False

# =========================================================
# 3. 민감 정보 (.env에서 호출)
# =========================================================
# Naver API
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# Database
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME")
}

# Azure ML
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")