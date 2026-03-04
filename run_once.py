from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
import config # [추가] 환경 설정 임포트

COMPUTE_NAME = "Data-Collect-Pipeline"

# 1. Azure ML 연결
credential = DefaultAzureCredential()
# [수정] config 모듈 변수 호출
ml_client = MLClient(
    credential,
    subscription_id=config.AZURE_SUBSCRIPTION_ID,
    resource_group_name=config.AZURE_RESOURCE_GROUP,
    workspace_name=config.AZURE_WORKSPACE_NAME
)

print(f"✅ Azure ML 연결 성공. 수동 작업을 제출합니다...")

# 2. 1회성 작업(Job) 정의
job = command(
    code="./",
    command="python pipeline.py",
    environment=Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
        conda_file="./conda_env.yaml",
    ),
    compute=COMPUTE_NAME,
    display_name="manual-test-run",
    experiment_name="pipeline-manual-test"
)

# 3. 작업 제출 및 로그 스트리밍
try:
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"🚀 작업이 제출되었습니다! 상태: {returned_job.status}")
    print(f"🔗 작업 상세 페이지: {returned_job.studio_url}")
    print("⏳ 클러스터 준비 및 실행 대기 중... (로그 스트리밍 시작)")
    
    ml_client.jobs.stream(returned_job.name)
    
except Exception as e:
    print(f"❌ 작업 제출 실패: {e}")