import pandas as pd
import io
import csv
import urllib.parse
import json
from datetime import datetime
from sqlalchemy import create_engine, text
import numpy as np
import config  # [추가] 환경 설정 및 DB 정보 임포트

def format_vector_for_pg(vec):
    """
    [벡터 포맷팅]
    """
    if vec is None: return None
    if isinstance(vec, float) and np.isnan(vec): return None
    if isinstance(vec, (list, np.ndarray)):
        if len(vec) == 0: return None
        return json.dumps(list(vec))
    return str(vec)

def sync_pk_sequence(engine):
    """
    [PK 시퀀스 동기화]
    """
    try:
        with engine.connect() as conn:
            sql = text("""
                SELECT setval(
                    pg_get_serial_sequence('tbl_collect_trademark', 'c_trademark_no'), 
                    COALESCE((SELECT MAX(c_trademark_no) FROM tbl_collect_trademark), 0) + 1, 
                    false
                );
            """)
            conn.execute(sql)
            conn.commit()
            print("✅ [DB Handler] PK 시퀀스(번호표) 자동 동기화 완료")
    except Exception as e:
        print(f"⚠️ [DB Handler] 시퀀스 동기화 실패 (테이블 비어있음 등): {e}")

def load_queue_to_db(input_queue):
    print("🚀 [DB Handler] 적재 대기 중...")
    
    # [수정] config에서 DB 정보 가져오기
    safe_password = urllib.parse.quote_plus(config.DB_CONFIG['password'])
    conn_url = f"postgresql+psycopg2://{config.DB_CONFIG['user']}:{safe_password}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['dbname']}"
    engine = create_engine(conn_url)
    
    sync_pk_sequence(engine)
    
    total_inserted = 0

    while True:
        batch_items = input_queue.get()
        if batch_items is None:
            print(f"✅ [DB Handler] 적재 완료. 총 {total_inserted}건.")
            break
            
        df = pd.DataFrame(batch_items)
        
        col_map = {
            'title': 'c_product_name',
            'link': 'c_product_page_url',
            'mallName': 'c_manufacturer_info',
            'brand': 'c_brand_info',
            'category1': 'c_l_category',
            'category2': 'c_m_category',
            'category3': 'c_s_category',
            'TradeMark_code': 'c_trademark_class_code',
            'image_b64': 'c_trademark_image',
            'ocr_text': 'c_trademark_name',
            'c_trademark_type': 'c_trademark_type'
        }
        
        for c in col_map.keys():
            if c not in df.columns: df[c] = None
            
        df = df.rename(columns=col_map)
        
        df['c_trademark_ent_date'] = datetime.now()
        
        if 'c_trademark_name_vec' in df.columns:
            df['c_trademark_name_vec'] = df['c_trademark_name_vec'].apply(format_vector_for_pg)
        else:
            df['c_trademark_name_vec'] = None

        if 'c_trademark_image_vec' in df.columns:
            df['c_trademark_image_vec'] = df['c_trademark_image_vec'].apply(format_vector_for_pg)
        else:
            df['c_trademark_image_vec'] = None

        db_cols = [
            'c_product_name', 'c_product_page_url', 'c_manufacturer_info', 
            'c_brand_info', 'c_l_category', 'c_m_category', 'c_s_category',
            'c_trademark_type', 'c_trademark_class_code', 'c_trademark_name',
            'c_trademark_name_vec',
            'c_trademark_image', 
            'c_trademark_image_vec',
            'c_trademark_ent_date'
        ]
        
        for col in db_cols:
            if col not in df.columns: df[col] = None

        output = io.StringIO()
        df[db_cols].to_csv(output, sep='\t', header=False, index=False, quoting=csv.QUOTE_MINIMAL, doublequote=True, escapechar='\\')
        output.seek(0)
        
        try:
            raw_conn = engine.raw_connection()
            cur = raw_conn.cursor()
            cols_str = ', '.join(db_cols)
            
            sql = f"COPY tbl_collect_trademark ({cols_str}) FROM STDIN WITH (FORMAT CSV, DELIMITER '\t', NULL '')"
            cur.copy_expert(sql, output)
            
            raw_conn.commit()
            cur.close()
            raw_conn.close()
            
            total_inserted += len(df)
            print(f"⚡ [DB] {len(df)}건 적재 완료 (벡터 포함)")
            
        except Exception as e:
            print(f"❌ DB 적재 에러: {e}")