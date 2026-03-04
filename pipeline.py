from prefect import flow
import threading
from queue import Queue
from tasks.data_fetcher import fetch_data_to_queue
from tasks.ai_processor import process_queue_data
from tasks.db_handler import load_queue_to_db
import config

check_embed_mode = lambda: "False" if config.USE_DUMMY_TEXT_EMBEDDING else "True"

@flow(name="Modular-Streaming-Pipeline", log_prints=True)
def shopping_flow():
    print("--- [시작] 동시성 스트리밍 파이프라인 ---")
    print("파이프라인 1.0 v")
    print(f"실제 텍스트 임베딩 여부: {check_embed_mode()}")
    
    # 데이터 통로(Queue) 생성 (메모리 버퍼 역할)
    fetch_queue = Queue(maxsize=100) 
    db_queue = Queue(maxsize=100)
    
    # 3개의 작업자(Thread) 생성 (config에서 네이버 키 호출)
    t1 = threading.Thread(target=fetch_data_to_queue, args=(fetch_queue, config.NAVER_CLIENT_ID, config.NAVER_CLIENT_SECRET))
    t2 = threading.Thread(target=process_queue_data, args=(fetch_queue, db_queue))
    t3 = threading.Thread(target=load_queue_to_db, args=(db_queue,))
    
    # 동시 실행 시작
    t1.start()
    t2.start()
    t3.start()
    
    # 모두 끝날 때까지 대기
    t1.join()
    t2.join()
    t3.join()
    
    print("--- [종료] 모든 파이프라인 처리 완료 ---")

if __name__ == "__main__":
    shopping_flow()