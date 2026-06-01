import cv2
import os
import glob
import time
import numpy as np
from ultralytics import YOLO

# [경로 설정] 본인 환경에 맞게 지정
DAVIS_ROOT = "./DAVIS"
TEST_DATA_DIR = "./yolo_davis_dataset/images/test"

# 스크린샷에 찍힌 경로 기준 가중치 파일 설정
RUN1_WEIGHTS = "./runs/detect/ml_project/yolo_davis_run1/weights/best.pt"
RUN2_WEIGHTS = "./runs/detect/ml_project/yolo_davis_run2/weights/best.pt"

def mask_to_bbox(mask_path):
    """흑백 마스크를 [x, y, w, h] 바운딩 박스로 변환"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    return list(cv2.boundingRect(largest_contour))

def calculate_iou(box1, box2):
    """두 박스의 IoU 계산"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_test_sequences():
    """Test 폴더 내 파일명을 분석하여 배정된 Test 시퀀스 목록 추출"""
    if not os.path.exists(TEST_DATA_DIR):
        return []
    files = os.listdir(TEST_DATA_DIR)
    sequences = set()
    for f in files:
        if f.endswith(".jpg"):
            # '시퀀스명_프레임번호.jpg' 구조에서 시퀀스명 추출
            parts = f.rsplit("_", 1)
            if len(parts) > 0:
                sequences.add(parts[0])
    return sorted(list(sequences))

def visualize_model(model_path, seq_name, output_filename, model_label):
    """특정 모델을 이용해 비디오 시각화 렌더링 및 최종 성능 측정"""
    model = YOLO(model_path)
    
    img_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", seq_name)
    mask_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq_name)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    if not img_files: return None, None
    
    # 비디오 인코더 설정
    first_frame = cv2.imread(img_files[0])
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_filename, fourcc, 24.0, (w, h))
    
    iou_list, fps_list = [], []
    
    for img_path, mask_path in zip(img_files, mask_files):
        frame = cv2.imread(img_path)
        if frame is None: continue
        current_gt = mask_to_bbox(mask_path)
        
        # 모델 추적 및 시간 측정
        start_time = time.time()
        results = model.track(frame, persist=True, verbose=False)
        fps = 1.0 / (time.time() - start_time)
        
        box_pred = [0, 0, 0, 0]
        success = False
        if results[0].boxes and len(results[0].boxes) > 0:
            xyxy = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            box_pred = [x1, y1, x2 - x1, y2 - y1]
            success = True
            
        iou = calculate_iou(box_pred, current_gt) if current_gt else 0.0
        iou_list.append(iou)
        fps_list.append(fps)
        
        # 🎨 시각화 그리기
        # A. Ground Truth (정답) -> 녹색
        if current_gt:
            gx, gy, gw, gh = current_gt
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            cv2.putText(frame, "GT", (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # B. Prediction (모델 예측) -> 적색
        if success:
            px, py, pw, ph = box_pred
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(frame, model_label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # C. 스코어 보드 배경 렌더링
        cv2.rectangle(frame, (10, 10), (240, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"IoU: {iou:.4f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_video.write(frame)
        
    out_video.release()
    return np.mean(iou_list), np.mean(fps_list)

def main():
    print("🔍 Test 세트에 배정된 비디오 시퀀스를 탐색합니다...")
    test_seqs = get_test_sequences()
    
    if not test_seqs:
        print("❌ Test 폴더에서 영상을 찾을 수 없습니다. 전처리가 완료되었는지 확인하세요.")
        return
        
    print(f"✨ 발견된 Test 시퀀스 목록: {test_seqs}")
    
    # 첫 번째로 발견된 Test 영상을 평가 대상으로 선택 (예: 'blackswan' 등)
    selected_seq = test_seqs[0]
    print(f"🎯 최종 시각화 평가 대상 시퀀스 결정: [{selected_seq}]")
    print("="*50)
    
    # 1. Run1 (SGD 설정) 검증 및 동영상 추출
    if os.path.exists(RUN1_WEIGHTS):
        print(f"🏋️‍♂️ [실험 1: SGD] 가중치 기반 테스트 비디오 생성 중...")
        r1_out = f"./test_vizo_run1_{selected_seq}.mp4"
        r1_iou, r1_fps = visualize_model(RUN1_WEIGHTS, selected_seq, r1_out, "YOLOv8_Run1_SGD")
        print(f"✅ Run1 비디오 저장 완료 -> {r1_out}")
        print(f"📊 [Run1 성적] 평균 IoU: {r1_iou:.4f} | 평균 FPS: {r1_fps:.2f}\n")
    else:
        print(f"❌ Run1 가중치 파일을 찾을 수 없습니다: {RUN1_WEIGHTS}")

    # 2. Run2 (AdamW 설정) 검증 및 동영상 추출
    if os.path.exists(RUN2_WEIGHTS):
        print(f"🏋️‍♂️ [실험 2: AdamW] 가중치 기반 테스트 비디오 생성 중...")
        r2_out = f"./test_vizo_run2_{selected_seq}.mp4"
        r2_iou, r2_fps = visualize_model(RUN2_WEIGHTS, selected_seq, r2_out, "YOLOv8_Run2_AdamW")
        print(f"✅ Run2 비디오 저장 완료 -> {r2_out}")
        print(f"📊 [Run2 성적] 평균 IoU: {r2_iou:.4f} | 평균 FPS: {r2_fps:.2f}\n")
    else:
        print(f"❌ Run2 가중치 파일을 찾을 수 없습니다: {RUN2_WEIGHTS}")
        
    print("="*50)
    print("🎉 모든 시각화 테스트 비디오가 정상 추출되었습니다. 보고서 장표에 수치를 기입하세요.")

if __name__ == "__main__":
    main()