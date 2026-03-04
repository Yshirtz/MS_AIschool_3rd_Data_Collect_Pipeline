from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import JobSchedule, RecurrenceTrigger, PipelineJob, RecurrencePattern
from azure.identity import DefaultAzureCredential
import config # [추가] 환경 설정 임포트

COMPUTE_NAME = "Data-Collect-Pipeline"

credential = DefaultAzureCredential()
# [수정] config 모듈 변수 호출
ml_client = MLClient(
    credential, 
    subscription_id=config.AZURE_SUBSCRIPTION_ID, 
    resource_group_name=config.AZURE_RESOURCE_GROUP, 
    workspace_name=config.AZURE_WORKSPACE_NAME
)
print(f"✅ Azure ML 연결 성공: {ml_client.workspace_name}")

env_name = "pipeline-env-custom"
env_version = "1.0"

step_command = command(
    code="./", 
    command="python pipeline.py", 
    environment=f"{env_name}:{env_version}",
    compute=COMPUTE_NAME, 
    display_name="run-pipeline-script",
    is_deterministic=False 
)

pipeline_job = PipelineJob(
    display_name="daily-pipeline-wrapper",
    description="Daily Data Collection Pipeline",
    jobs={
        "main_worker": step_command 
    }
)

schedule_name = "daily-data-collection"

trigger = RecurrenceTrigger(
    frequency="Day",
    interval=1,
    schedule=RecurrencePattern(
        hours=[0],
        minutes=[0]
    )
)

schedule = JobSchedule(
    name=schedule_name,
    trigger=trigger,
    create_job=pipeline_job,
    description="매일 오전 9시(KST)에 실행되는 데이터 수집 파이프라인"
)

print(f"🚀 스케줄 '{schedule_name}' 시간 업데이트(09:00 KST)를 시작합니다...")

poller = ml_client.schedules.begin_create_or_update(schedule)
created_schedule = poller.result()

print(f"✅ 스케줄 등록 성공!")
print(f"📅 상태: {created_schedule.provisioning_state}")
print(f"⏰ 실행 시간 설정: 매일 UTC 00:00 (한국 시간 09:00)")
print(f"🔗 확인 링크: https://ml.azure.com/schedules?wsid={ml_client.workspace_name}")