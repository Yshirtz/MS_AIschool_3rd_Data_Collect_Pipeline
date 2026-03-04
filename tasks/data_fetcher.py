import time
import requests
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import config  # [추가] 환경 설정 및 DB 정보 임포트

# =========================================================
# 1. 통신 설정 (Session)
# =========================================================
def create_session():
    """
    [통신 최적화]
    매번 연결을 새로 맺지 않고(Handshake 생략), 
    연결을 유지(Keep-Alive)하여 API 호출 속도를 높입니다.
    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
    session.mount('https://', adapter)
    return session

GLOBAL_SESSION = create_session()

# =========================================================
# 2. 헬퍼 함수 (필터링)
# =========================================================
def is_own_trademark_product(item, p_trademark_name):
    """
    [브랜드 필터링 - 자사 정품 제외]
    """
    if not p_trademark_name: return False
    
    target_trademark = str(p_trademark_name).strip().lower()
    
    api_brand = item.get('brand', '')
    if api_brand:
        api_brand = str(api_brand).strip().lower()
    
    if api_brand and target_trademark == api_brand:
        return True
    
    api_maker = item.get('maker', '')
    if api_maker:
        api_maker = str(api_maker).strip().lower()
        
    if api_maker and target_trademark == api_maker:
        return True
        
    return False

def load_existing_urls(engine):
    """
    [중복 방지 1단계]
    """
    print("⏳ DB에서 기존 적재된 상품 URL 목록을 로딩 중입니다...")
    try:
        query = "SELECT c_product_page_url FROM tbl_collect_trademark"
        df = pd.read_sql(query, con=engine)
        existing_urls = set(df['c_product_page_url'].dropna().tolist())
        print(f"📋 기존 DB URL 로딩 완료: 총 {len(existing_urls)}건")
        return existing_urls
    except Exception as e:
        print(f"⚠️ 기존 URL 로딩 실패: {e}")
        return set()

# =========================================================
# 3. 메인 수집 로직
# =========================================================
def fetch_data_to_queue(output_queue, client_id, client_secret):
    """
    [데이터 수집가 메인 함수]
    """
    print("🚀 [Fetcher] 수집 준비 시작 (화이트리스트 OFF)...")
    
    try:
        # [수정] config에서 DB 정보 가져오기
        safe_password = urllib.parse.quote_plus(config.DB_CONFIG['password'])
        conn_url = f"postgresql+psycopg2://{config.DB_CONFIG['user']}:{safe_password}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['dbname']}"
        engine = create_engine(conn_url)
        
        # 1. 중복 필터링용 데이터 로드
        existing_db_urls = load_existing_urls(engine)

        # 2. 검색 대상(Target) 조회 쿼리
        query = """
            SELECT DISTINCT 
                a.product_name as search_keyword, 
                CAST(a.class_code AS INTEGER) as product_code,
                b.p_trademark_name as trademark_name
            FROM tbl_p_trademark_product a
            JOIN tbl_protection_trademark b ON a.p_trademark_reg_no = b.p_trademark_reg_no
            JOIN tbl_product_code c ON a.product_name = c.product_name
            WHERE b.manage_end_date IS NULL  -- 유효한 상표권만
              AND a.class_code < '35'        -- 상품류(Goods)만 대상 (서비스업 제외)
        """
        targets = pd.read_sql(query, con=engine)
    except Exception as e:
        print(f"[Fetcher] DB 초기화 실패: {e}")
        output_queue.put(None)
        return

    # 네이버 API 헤더 설정
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    url = "https://openapi.naver.com/v1/search/shop.json"
    
    total_fetched = 0
    cnt_skip_db = 0
    cnt_skip_local = 0
    cnt_skip_brand = 0
    
    seen_product_ids = set()

    print(f"🚀 [Fetcher] 수집 시작 (타겟: {len(targets)}개 키워드)...")

    for _, row in targets.iterrows():
        keyword = row['search_keyword']
        p_code = row['product_code']
        trademark_name = row['trademark_name']
        
        for start_idx in range(1, 1001, 100):
            params = {
                "query": keyword, 
                "display": 100,
                "start": start_idx, 
                "sort": "date"
            }
            try:
                resp = GLOBAL_SESSION.get(url, headers=headers, params=params, timeout=3)
                if resp.status_code != 200: break
                
                data = resp.json()
                items = data.get('items', [])
                if not items: break
                
                batch_items = []
                
                for item in items:
                    p_id = str(item.get('productId'))
                    p_url = item.get('link')

                    if p_id in seen_product_ids:
                        cnt_skip_local += 1
                        continue
                    
                    if p_url in existing_db_urls:
                        cnt_skip_db += 1
                        continue

                    if is_own_trademark_product(item, trademark_name):
                        cnt_skip_brand += 1
                        continue

                    seen_product_ids.add(p_id)
                    item['TradeMark_code'] = p_code
                    
                    if "title" in item:
                        item["title"] = item["title"].replace("<b>", "").replace("</b>", "")
                    
                    batch_items.append(item)
                
                if batch_items:
                    output_queue.put(batch_items)
                    total_fetched += len(batch_items)
                    
            except Exception:
                continue
                
    print("="*50)
    print(f"✅ [Fetcher] 수집 완료")
    print(f"   - 수집 성공 및 전송: {total_fetched}건")
    print(f"   - 중복 제외(DB+Local): {cnt_skip_db + cnt_skip_local}건")
    print(f"   - 브랜드 일치(정품) 제외: {cnt_skip_brand}건")
    print("="*50)
    
    output_queue.put(None)