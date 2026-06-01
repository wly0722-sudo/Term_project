import cv2
import os
import glob
import time
import numpy as np

# [경로 및 시퀀스 설정]
DAVIS_ROOT = "./DAVIS"
SEQ_NAME = "bmx-bumps"  # 🎯 대조군 장표용 격렬한 자전거 시퀀스 고정
OUTPUT_VIDEO = f"./traditional_dcf_failure_{SEQ_NAME}.mp4"

def mask_to_bbox(mask_path):
    """흑백 마스크에서 [x, y, w, h] 정답 바운딩 박스 추출"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    return list(cv2.boundingRect(largest_contour))

def calculate_iou(box1, box2):
    """두 박스의 IoU 계산 (format: [x, y, w, h])"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main():
    img_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", SEQ_NAME)
    mask_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", SEQ_NAME)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    if not img_files:
        print(f"❌ [{SEQ_NAME}] 시퀀스 경로를 찾을 수 없습니다. DAVIS 폴더 구조를 확인하세요.")
        return

    print(f"🎬 전통적 OpenCV CSRT 추적기 시뮬레이션을 시작합니다. 대상: [{SEQ_NAME}]")
    
    # 1. OpenCV CSRT 추적기 인스턴스 생성
    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        tracker = cv2.legacy.TrackerCSRT_create()
        
    # 2. 첫 번째 프레임(Index 0) 로드 및 추적기 초기화
    first_img = cv2.imread(img_files[0])
    h, w, _ = first_img.shape
    init_bbox = mask_to_bbox(mask_files[0])
    
    if init_bbox is None:
        print("❌ 첫 번째 프레임에서 마스크 정답 박스를 추출하지 못했습니다.")
        return
        
    tracker.init(first_img, tuple(init_bbox))
    
    # 비디오 라이터 및 성적 저장 리스트 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 24.0, (w, h))
    
    iou_list = []
    fps_list = []
    
    print("🔄 프레임 순회 및 실시간 IoU/FPS 연산 중...")
    
    # 3. 0번째 프레임을 건너뛰고 1번째 프레임부터 루프 수행 (⚠️ 튕김 원인 원천 차단 패치)
    for i in range(1, len(img_files)):
        frame = cv2.imread(img_files[i])
        if frame is None: continue
        
        # Ground Truth (초록색 박스)
        current_gt = mask_to_bbox(mask_files[i])
        
        # ⏱️ CSRT 추적 및 시간 측정
        start_time = time.time()
        success, pred_box = tracker.update(frame)
        elapsed = time.time() - start_time
        
        # FPS 환산 및 저장
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_list.append(fps)
        
        # IoU 계산 및 저장
        iou = 0.0
        box_pred = list(pred_box) if success else [0, 0, 0, 0]
        if current_gt:
            iou = calculate_iou(box_pred, current_gt)
            iou_list.append(iou)
        
        # 🎨 시각화 레이어 렌더링
        # A. Ground Truth -> 녹색
        if current_gt:
            gx, gy, gw, gh = current_gt
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.putText(frame, "GT", (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # B. OpenCV CSRT Prediction -> 적색
        if success:
            px, py, pw, ph = map(int, box_pred)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(frame, "OpenCV CSRT", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # C. 좌측 상단 실시간 스코어 보드
        cv2.rectangle(frame, (10, 10), (240, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"IoU: {iou:.4f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out_video.write(frame)
        
    out_video.release()
    
    # 📊 최종 대조 평가지표 도출
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    mean_fps = np.mean(fps_list) if fps_list else 0.0
    
    print("="*50)
    print(f"🎉 전통적 CV 시뮬레이션 완료! -> {OUTPUT_VIDEO}")
    print(f"📊 [CSRT 최종 성적] 대상 시퀀스: {SEQ_NAME}")
    print(f"📈 평균 IoU: {mean_iou:.4f}")
    print(f"⚡ 평균 FPS: {mean_fps:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()