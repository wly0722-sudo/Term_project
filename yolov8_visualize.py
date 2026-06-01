import cv2
import os
import glob
import time
import numpy as np
from ultralytics import YOLO

# [경로 설정] 본인의 DAVIS 폴더 이름과 맞추세요.
DAVIS_ROOT = "./DAVIS"
SEQUENCE = "bmx-bumps"
OUTPUT_VIDEO_PATH = "./yolo_proposed_visualization.mp4"

def mask_to_bbox(mask_path):
    """흑백 마스크 이미지를 바운딩 박스 [x, y, w, h]로 변환"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    return list(cv2.boundingRect(largest_contour))

def calculate_iou(box1, box2):
    """두 박스의 가중치 겹침 비율(IoU) 계산"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def run_yolo_pipeline():
    img_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", SEQUENCE)
    mask_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", SEQUENCE)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not img_files:
        print(f"❌ 영상을 찾을 수 없습니다. 경로를 확인하세요: {img_dir}")
        return

    # 1. YOLOv8 사전 학습 모델 로드
    model = YOLO("yolov8n.pt")
    
    first_frame = cv2.imread(img_files[0])
    frame_h, frame_w, _ = first_frame.shape

    # 2. 비디오 인코더 설정 (원본 16:9 크기 지정)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 24.0, (frame_w, frame_h))

    print("🚀 YOLOv8 Proposed 시각화 파이프라인 시작...")
    iou_list, fps_list = [], []

    for i in range(1, len(img_files)):
        frame = cv2.imread(img_files[i])
        if frame is None: continue
        current_gt = mask_to_bbox(mask_files[i])

        # --- YOLO Tracking 실행 및 FPS 측정 ---
        start_time = time.time()
        yolo_results = model.track(frame, persist=True, verbose=False)
        fps = 1.0 / (time.time() - start_time)

        box_pred = [0, 0, 0, 0]
        success = False
        
        # YOLO 결과에서 박스 좌표 파싱
        if yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
            xyxy = yolo_results[0].boxes.xyxy[0].cpu().numpy()
            x1_y, y1_y, x2_y, y2_y = map(int, xyxy)
            box_pred = [x1_y, y1_y, x2_y - x1_y, y2_y - y1_y]
            success = True

        # --- 평가 지표 계산 ---
        iou = 0.0
        if current_gt:
            iou = calculate_iou(box_pred, current_gt)
            iou_list.append(iou)
            fps_list.append(fps)

        # ---------------------------------------------------------
        # 🎨 시각화 레이어 추가 (영상 위에 직접 그리기)
        # ---------------------------------------------------------
        # A. Ground Truth (정답 박스) -> 녹색 (Green)
        if current_gt:
            gx, gy, gw, gh = current_gt
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.putText(frame, "GT (Ground Truth)", (gx, gy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # B. Model Prediction (예측 박스) -> 적색 (Red)
        if success:
            px, py, pw, ph = box_pred
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(frame, "YOLOv8 Object Track", (px, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # C. 좌측 상단 실시간 스코어 보드
        cv2.rectangle(frame, (10, 10), (220, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"IoU: {iou:.4f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 박스가 그려진 최종 프레임을 비디오 파일에 저장
        out_video.write(frame)

    out_video.release()
    print(f"✅ YOLOv8 시각화 비디오 저장 완료! -> {OUTPUT_VIDEO_PATH}")
    print(f"📊 평균 IoU: {np.mean(iou_list):.4f} | 평균 FPS: {np.mean(fps_list):.2f}")

if __name__ == "__main__":
    run_yolo_pipeline()