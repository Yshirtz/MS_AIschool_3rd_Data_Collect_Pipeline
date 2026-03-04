from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
import config # [추가] 환경 설정 임포트

COMPUTE_NAME = "Data-Collect-Pipeline"

# ---------------------------------------------------------
# Azure 연결
# ---------------------------------------------------------
credential = DefaultAzureCredential()
# [수정] config 모듈 변수 호출
ml_client = MLClient(credential, config.AZURE_SUBSCRIPTION_ID, config.AZURE_RESOURCE_GROUP, config.AZURE_WORKSPACE_NAME)

print(f"Azure 연결 성공! 클러스터 '{COMPUTE_NAME}' 생성을 시도합니다...")

try:
    cluster = ml_client.compute.get(COMPUTE_NAME)
    print(f"✅ 이미 존재하는 클러스터입니다: {COMPUTE_NAME} (상태: {cluster.state})")
except Exception:
    print("✨ 새로운 클러스터를 생성합니다...")
    
    compute_config = AmlCompute(
        name=COMPUTE_NAME,
        size="Standard_NC4as_T4_v3",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
        tier="Dedicated",
    )
    
    ml_client.compute.begin_create_or_update(compute_config).result()
    print(f"🚀 클러스터 생성 완료! 이름: {COMPUTE_NAME}")

print("이제 run_once.py나 create_schedule.py를 실행할 준비가 되었습니다.")