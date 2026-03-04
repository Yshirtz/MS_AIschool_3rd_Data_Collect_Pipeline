import os
import cv2
import torch
import numpy as np
import base64
import json
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO, SAM
from paddleocr import PaddleOCR
import logging

# [리팩터링] 분리한 모듈 임포트
import config
from .models import ResNet50TripletNet, resnet_preprocess
from .utils import (
    setup_gpu_paths, create_session, download_image, 
    convert_bgra_to_rgb, is_useful_text, run_async_wrapper
)

# 로깅 및 환경 설정
logging.getLogger("ppocr").setLevel(logging.WARNING)
load_dotenv()

# GPU 설정 (utils.py에서 불러옴)
setup_gpu_paths()

# =========================================================
# 전역 변수 및 설정
# =========================================================
DOWNLOAD_WORKERS = 16
BATCH_SIZE = 8
PNG_COMPRESSION = 1
torch.set_num_threads(4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 세션 생성 (utils.py 사용)
GLOBAL_SESSION = create_session(pool_size=DOWNLOAD_WORKERS)

# 모델 저장소
model_container = {
    "yolo": None,
    "sam": None,
    "resnet": None,
    "ocr": None
}

def init_models():
    """모델 로딩 (Lazy Loading)"""
    print(f"[INFO] AI Processor Device: {DEVICE}")

    # 1. YOLO
    if model_container["yolo"] is None:
        print(f"[INFO] YOLO 로딩: {config.YOLO_MODEL_PATH}")
        model_container["yolo"] = YOLO(config.YOLO_MODEL_PATH)

    # 2. SAM
    if model_container["sam"] is None:
        print(f"[INFO] SAM 로딩: {config.SAM_MODEL_NAME}")
        model_container["sam"] = SAM(config.SAM_MODEL_NAME)

    # 3. ResNet (이미지 임베딩)
    if model_container["resnet"] is None:
        print(f"[INFO] ResNet 로딩: {config.RESNET_MODEL_PATH}")
        try:
            # models.py의 클래스 사용
            net = ResNet50TripletNet(embedding_dim=128).to(DEVICE)
            
            if not os.path.exists(config.RESNET_MODEL_PATH):
                raise FileNotFoundError(f"파일 없음: {config.RESNET_MODEL_PATH}")

            state_dict = torch.load(config.RESNET_MODEL_PATH, map_location=DEVICE)
            net.load_state_dict(state_dict)
            net.eval()
            model_container["resnet"] = net
            print("✅ ResNet 모델 로드 성공!")
        except Exception as e:
            print(f"❌ ResNet 로딩 실패: {e}")

    # 4. PaddleOCR
    if model_container["ocr"] is None:
        print(f"[INFO] OCR 로딩 (PaddleOCR)")
        try:
            model_container["ocr"] = PaddleOCR(
                use_angle_cls=True, 
                lang='korean', 
                det_limit_side_len=1200, 
                show_log=False
            )
            print("✅ OCR 로드 성공!")
        except Exception as e:
            print(f"❌ OCR 로딩 실패: {e}")

# =========================================================
# 기능 함수
# =========================================================

async def get_jina_embeddings_async(texts):
    """텍스트 임베딩 (더미 모드 지원)"""
    if not texts: return []
    
    # [더미 모드 체크] config.py 설정 확인
    if getattr(config, 'USE_DUMMY_TEXT_EMBEDDING', False):
        return [None] * len(texts)
    
    api_key = os.getenv("JINA_EMBED_API_KEY")
    url = os.getenv("JINA_EMBED_ENDPOINT")

    if not api_key or not url: return [None] * len(texts)

    valid_texts = [t if t else " " for t in texts]
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    payload = {"input": valid_texts, "model": "jina-embeddings-v3"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    result_json = await resp.json()
                    if "data" in result_json:
                        return [item["embedding"] for item in result_json["data"]]
    except: pass
    return [None] * len(texts)

def get_image_embeddings(pil_images):
    """ResNet 이미지 임베딩"""
    if not pil_images: return []
    model = model_container["resnet"]
    if model is None: return [None] * len(pil_images)

    try:
        # models.py의 전처리 함수 사용
        batch_tensors = []
        for img in pil_images:
            if img.mode != 'RGB': img = img.convert('RGB')
            batch_tensors.append(resnet_preprocess(img))
        
        if not batch_tensors: return [None] * len(pil_images)

        input_batch = torch.stack(batch_tensors).to(DEVICE)
        
        with torch.no_grad():
            embeddings = model(input_batch)
            
        return embeddings.cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ Image Embedding Error: {e}")
        return [None] * len(pil_images)

def get_ocr_text(processed_images):
    default_results = [""] * len(processed_images)
    ocr_reader = model_container["ocr"]
    if not ocr_reader: return default_results

    ocr_results = []
    for img in processed_images:
        try:
            result = ocr_reader.ocr(img, cls=True)
            extracted_text = ""
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line[1][1] > 0.6]
                extracted_text = " ".join(texts).strip()
            ocr_results.append(extracted_text)
        except:
            ocr_results.append("")
    return ocr_results

def process_batch_inference(image_list):
    """YOLO + SAM 추론"""
    if not image_list: return [], []
    
    b64_results = [None] * len(image_list)
    processed_imgs = [None] * len(image_list)
    
    yolo = model_container["yolo"]
    sam = model_container["sam"]
    if not yolo or not sam: return [], []

    try:
        # [수정됨] YOLO 기준 완화 (0.45 -> 0.25)
        # 이제 웬만한 물체는 다 잡아서 다음 단계로 넘깁니다.
        yolo_outputs = yolo.predict(image_list, conf=0.25, iou=config.IOU_THRESHOLD, verbose=False, device=DEVICE)
        
        for i, yolo_res in enumerate(yolo_outputs):
            img = image_list[i]
            if len(yolo_res.boxes) == 0: continue
            
            # Box Filtering
            valid_bboxes = []
            for box in yolo_res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                if (x2 - x1) < 40 or (y2 - y1) < 40: continue
                valid_bboxes.append([x1, y1, x2, y2])
            
            if not valid_bboxes: continue
            
            # SAM Inference
            try:
                sam_res = sam(img, bboxes=np.array(valid_bboxes), verbose=False, device=DEVICE)
                if sam_res[0].masks is None: continue
                
                # Mask Combination
                masks = sam_res[0].masks.data.cpu().numpy()
                h, w = img.shape[:2]
                combined_mask = np.zeros((h, w), dtype=bool)
                for mask in masks:
                    if mask.shape != (h, w): 
                        mask = cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)
                    combined_mask = np.logical_or(combined_mask, mask)
                
                # Background Removal
                final_result = np.zeros((h, w, 4), dtype=np.uint8)
                img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                final_result[combined_mask] = img_bgra[combined_mask]
                
                # Encoding
                success, encoded_buffer = cv2.imencode('.png', final_result, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
                if success:
                    b64_results[i] = f"\\x{encoded_buffer.tobytes().hex()}"
                processed_imgs[i] = final_result
            except: continue
            
    except Exception as e:
        print(f"Inference Error: {e}")
    
    return b64_results, processed_imgs

# =========================================================
# 메인 프로세서 (Data Pipeline)
# =========================================================
def process_queue_data(input_queue, output_queue):
    print("🚀 [AI Processor] 모델 초기화 중...")
    init_models()
    print(f"🚀 [AI Processor] 가동 시작 (Batch: {BATCH_SIZE})")
    
    while True:
        batch_items = input_queue.get()
        if batch_items is None:
            output_queue.put(None)
            break
            
        img_urls = [item.get('image') for item in batch_items]
        
        # 이미지 병렬 다운로드 (Session 전달)
        download_args = [(GLOBAL_SESSION, url) for url in img_urls]
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as io_executor:
            downloaded_images = list(io_executor.map(download_image, download_args))
        
        # [수정됨] Numpy 배열 에러 방지 (count(None) 대신 sum 사용)
        total_items_count = len(batch_items)
        download_fail_count = sum(1 for img in downloaded_images if img is None)

        if download_fail_count > 0:
            print(f"⚠️ [Data Loss] 다운로드 실패: {download_fail_count}건 (링크 만료 또는 서버 에러)")

        processed_batch = []
        total_count = len(batch_items)
        
        # YOLO 인식 실패 카운터
        yolo_fail_total = 0 
        
        for i in range(0, total_count, BATCH_SIZE):
            chunk_items = batch_items[i : i + BATCH_SIZE]
            chunk_imgs  = downloaded_images[i : i + BATCH_SIZE]
            
            valid_inputs = []
            valid_indices = [] 
            
            for idx, img in enumerate(chunk_imgs):
                if img is not None:
                    valid_inputs.append(img)
                    valid_indices.append(idx)
            
            if not valid_inputs: continue
            
            # 1. Vision Inference (YOLO -> SAM)
            b64_results, processed_cv_imgs = process_batch_inference(valid_inputs)
            
            # 이 배치에서 YOLO가 버린 개수 계산
            current_yolo_fails = b64_results.count(None)
            yolo_fail_total += current_yolo_fails

            # 성공한 결과만 추리기
            success_local_indices = []
            target_ocr_imgs = []
            target_embed_imgs = []
            
            for k, (b64, proc_img) in enumerate(zip(b64_results, processed_cv_imgs)):
                if b64 and proc_img is not None:
                    rgb_img_np = convert_bgra_to_rgb(proc_img)
                    pil_img = Image.fromarray(rgb_img_np)
                    
                    target_ocr_imgs.append(rgb_img_np)
                    target_embed_imgs.append(pil_img)
                    success_local_indices.append(k)

            if not success_local_indices: continue

            # 2. OCR Extraction
            final_texts = get_ocr_text(target_ocr_imgs)
            
            # 3. Embeddings
            txt_vecs = run_async_wrapper(get_jina_embeddings_async(final_texts))
            img_vecs = get_image_embeddings(target_embed_imgs)
            
            # 4. Result Mapping
            for res_idx, local_k in enumerate(success_local_indices):
                original_idx = valid_indices[local_k]
                item = chunk_items[original_idx]
                
                item['image_b64'] = b64_results[local_k]
                item['c_trademark_image_vec'] = img_vecs[res_idx]

                raw_text = final_texts[res_idx]
                
                if is_useful_text(raw_text):
                    item['c_trademark_type'] = 'both'
                    item['ocr_text'] = raw_text
                    item['c_trademark_name_vec'] = txt_vecs[res_idx]
                else:
                    item['c_trademark_type'] = 'shape'
                    item['ocr_text'] = None
                    item['c_trademark_name_vec'] = None
                
                processed_batch.append(item)
        
        # 처리 완료 후 로그 출력
        if yolo_fail_total > 0:
            print(f"⚠️ [Data Loss] YOLO 인식 실패(물체 없음): {yolo_fail_total}건 (conf=0.25 기준)")

        if processed_batch:
            output_queue.put(processed_batch)
            print(f"📦 [AI Processor] 입력 {total_items_count}건 -> 처리 성공 {len(processed_batch)}건")
        else:
            print(f"⚠️ [AI Processor] 입력 {total_items_count}건 -> 처리 성공 0건 (전부 실패)")