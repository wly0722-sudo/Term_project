import cv2
import os
import glob
import time
import numpy as np
from ultralytics import YOLO

# [경로 설정]
DAVIS_ROOT = "./DAVIS"
TEST_DATA_DIR = "./yolo_davis_dataset/images/test"

# 💡 [변인 통제 핵심] 동일한 고정 가중치 파일 하나만 사용합니다.
FIXED_WEIGHTS = "./runs/detect/ml_project/yolo_davis_run1/weights/best.pt"

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
    """Test 폴더 내 배정된 시퀀스 목록 추출"""
    if not os.path.exists(TEST_DATA_DIR):
        return []
    files = os.listdir(TEST_DATA_DIR)
    sequences = set()
    for f in files:
        if f.endswith(".jpg"):
            parts = f.rsplit("_", 1)
            if len(parts) > 0:
                sequences.add(parts[0])
    return sorted(list(sequences))

def visualize_ablation(model_path, seq_name, output_filename, exp_label, conf_thresh, iou_thresh):
    """동일 모델 가중치 하에서 인퍼런스 파라미터만 다르게 하여 테스트"""
    model = YOLO(model_path)
    
    img_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p", seq_name)
    mask_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq_name)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    if not img_files: return None, None
    
    first_frame = cv2.imread(img_files[0])
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_filename, fourcc, 24.0, (w, h))
    
    iou_list, fps_list = [], []
    
    for img_path, mask_path in zip(img_files, mask_files):
        frame = cv2.imread(img_path)
        if frame is None: continue
        current_gt = mask_to_bbox(mask_path)
        
        start_time = time.time()
        results = model.track(
            frame, 
            persist=True, 
            verbose=False,
            conf=conf_thresh,
            iou=iou_thresh
        )
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
        
        # 🎨 시각화
        if current_gt:
            gx, gy, gw, gh = current_gt
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            
        if success:
            px, py, pw, ph = box_pred
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            cv2.putText(frame, f"{exp_label}", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # 스코어보드 오버레이
        cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"IoU: {iou:.4f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_video.write(frame)
        
    out_video.release()
    return np.mean(iou_list), np.mean(fps_list)

def main():
    if not os.path.exists(FIXED_WEIGHTS):
        print(f"❌ 가중치 파일을 찾을 수 없습니다. 경로를 확인하세요: {FIXED_WEIGHTS}")
        return

    print("🔍 Test 세트 시퀀스를 탐색합니다...")
    test_seqs = get_test_sequences()
    
    # 격렬한 움직임이 있는 'bmx-bumps' 시퀀스를 선택 (리스트에 맞게 인덱스 자동 지정)
    selected_seq = "bmx-bumps" if "bmx-bumps" in test_seqs else test_seqs[0]
    print(f"🎯 실험 대상 시퀀스 고정: [{selected_seq}]")
    print("="*50)
    
    # 🕵️‍♂️ 실험 A: 엄격한 가이드라인 (Baseline 파라미터)
    print(f"🏃‍♂️ [실험 A] Strict Setting (Conf 0.50 / NMS IoU 0.30) 연산 중...")
    out_a = f"./ablation_strict_{selected_seq}.mp4"
    iou_a, fps_a = visualize_ablation(FIXED_WEIGHTS, selected_seq, out_a, "Strict (Conf:0.5)", conf_thresh=0.5, iou_thresh=0.3)
    print(f"📊 [실험 A 결과] 평균 IoU: {iou_a:.4f} | 평균 FPS: {fps_a:.2f}\n")

    # 🕵️‍♂️ 실험 B: 유연한 가이드라인 (제안 파라미터)
    print(f"🏃‍♂️ [실험 B] Flexible Setting (Conf 0.25 / NMS IoU 0.60) 연산 중...")
    out_b = f"./ablation_flexible_{selected_seq}.mp4"
    iou_b, fps_b = visualize_ablation(FIXED_WEIGHTS, selected_seq, out_b, "Flexible (Conf:0.25)", conf_thresh=0.25, iou_thresh=0.6)
    print(f"📊 [실험 B 결과] 평균 IoU: {iou_b:.4f} | 평균 FPS: {fps_b:.2f}\n")
    
    print("="*50)
    print("🎉 동일 가중치 기반 인퍼런스 파라미터 소거 연구(Ablation Study)가 완료되었습니다.")

if __name__ == "__main__":
    main()