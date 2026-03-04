import os
import sys
import site
import re
import cv2
import numpy as np
import requests
import asyncio
import torch

# =========================================================
# 1. 시스템 및 GPU 설정
# =========================================================
def setup_gpu_paths():
    """Windows 환경에서 GPU 라이브러리 경로 강제 연결"""
    if os.name == 'nt':
        print("\n🛠️ [Init] GPU 라이브러리 경로 강제 주입 중...")
        try:
            torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
            
            site_packages = site.getsitepackages()
            for sp in site_packages:
                nvidia_dir = os.path.join(sp, 'nvidia')
                if os.path.exists(nvidia_dir):
                    for root, dirs, files in os.walk(nvidia_dir):
                        if any(f.lower().endswith('.dll') for f in files):
                            try:
                                os.add_dll_directory(root)
                            except: pass
            print("✅ [Init] GPU 라이브러리 경로 설정 완료")
        except Exception as e:
            print(f"⚠️ GPU 경로 설정 중 경고: {e}")

# =========================================================
# 2. 이미지 및 네트워크 유틸리티
# =========================================================
def create_session(pool_size=100):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def download_image(args):
    """
    ThreadPoolExecutor에서 사용하기 위해 인자를 튜플로 받거나, 
    partial을 사용할 수 있게 설계. 여기서는 간단히 (session, url) 튜플이나 url만 받게 처리.
    """
    # session과 url을 분리하는 로직 (호출부에서 session을 넘겨줄 경우)
    session, url = args if isinstance(args, tuple) else (None, args)
    
    if not url: return None
    try:
        # session이 없으면 requests 바로 사용 (비권장하지만 안전장치)
        req = session if session else requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = req.get(url, headers=headers, timeout=5) # 타임아웃도 3초 -> 5초로 늘림
        
        if response.status_code != 200: 
            print(f"네이버 쇼핑 서버에서 이미지 다운로드 중 뭔가 잘못됨. 에러코드 : {response.status_code}")
            return None
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def convert_bgra_to_rgb(image_cv2):
    if image_cv2 is None: return None
    if image_cv2.shape[2] == 4:
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGRA2RGB)
    else:
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

# =========================================================
# 3. 텍스트 및 기타 유틸리티
# =========================================================
def is_useful_text(text):
    if not text: return False
    text = text.strip()
    if len(text) == 0: return False
    if not re.search(r'[가-힣a-zA-Z0-9]', text):
        return False
    return True

def run_async_wrapper(coroutine):
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)