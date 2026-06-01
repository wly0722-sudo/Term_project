import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# =========================================================
# [파라미터 및 경로 제어 설정]
# =========================================================
INPUT_VIDEO_PATH = "./input_video.mp4"       # 🎞️ 변환하고 싶은 원본 와이드 영상 경로
OUTPUT_VIDEO_PATH = "./output_reframe_9_16.mp4" # 📱 결과 세로형 쇼츠 영상 경로

# 🏆 [변인 통제] 실험에서 가장 우수한 성능을 보인 최적 파라미터 강제 고정
MODEL_WEIGHTS = "./runs/detect/ml_project/yolo_davis_run2/weights/best.pt"
CONF_THRESH = 0.25
IOU_THRESH = 0.6   # NMS 임계값

def run_video_auto_reframe():
    # 0. 가중치 파일 존재 여부 예외 처리
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ [에러] 최적 가중치 가 존재하지 않습니다. 경로를 확인하세요: {MODEL_WEIGHTS}")
        return

    # 1. 비디오 인스턴스 캡처 및 메타데이터 파싱
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ [에러] 입력 비디오 파일을 로드할 수 없습니다: {INPUT_VIDEO_PATH}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. 제안 모델(YOLOv8n 파인튜닝 가중치) 아키텍처 로드
    model = YOLO(MODEL_WEIGHTS)
    
    # 3. 시공간 제약을 고려한 9:16 크롭 바운더리 규격 정의 (높이 고정 방식)
    crop_w = int(frame_h * (9 / 16))
    crop_h = frame_h
    
    # 출력 비디오 라이터 세팅
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (crop_w, crop_h))

    print(f"🚀 지능형 오토 리프레임(Auto Reframe) 엔지니어링 파이프라인 가동...")
    print(f"📊 인퍼런스 최적화 조건 -> Conf: {CONF_THRESH} | NMS IoU: {IOU_THRESH}")
    print(f"📐 해상도 전환 메트릭: {frame_w}x{frame_h} (16:9) ➡️ {crop_w}x{crop_h} (9:16)")
    print("-" * 60)
    
    last_cx = frame_w / 2  # 추적 단절(Occlusion) 발생 시 급격한 뷰 전환 방지용 앵커 변수
    frame_idx = 0

    # 4. 실시간 비디오 프레임 단위 스트림 순회
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 영상 종료 시 루프 탈출
            
        frame_idx += 1
        
        # --- YOLOv8 + ByteTRACK 기반 실시간 객체 인지 (가중치 파라미터 강제 주입) ---
        start_time = time.time()
        yolo_results = model.track(
            frame, 
            persist=True, 
            verbose=False,
            conf=CONF_THRESH,
            iou=IOU_THRESH
        )
        elapsed_time = time.time() - start_time
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

        box_pred = [0, 0, 0, 0]
        success = False
        
        # 최상위 신뢰도를 가진 타겟 객체의 2D 바운딩 박스 좌표 추출
        if yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
            xyxy = yolo_results[0].boxes.xyxy[0].cpu().numpy()
            x1_y, y1_y, x2_y, y2_y = map(int, xyxy)
            box_pred = [x1_y, y1_y, x2_y - x1_y, y2_y - y1_y]
            success = True

        # --- 동적 공간적 중심점(Dynamic Spatial Center) 연산 및 컨텍스트 보존 ---
        if success:
            cx = box_pred[0] + box_pred[2] / 2
            last_cx = cx  # 타겟이 정상 추적될 때 가중치 갱신
        else:
            cx = last_cx  # 순간적인 장애물 차폐 및 모션 블러 발생 시 이전 공간 맥락 유지 (지터링 방어)

        # 캔버스 경계면 예외 처리 (Boundary Clamping)
        # 박스가 화면 왼쪽 영점(0) 이하 혹은 오른쪽 해상도 최대치 이상으로 삐져나가는 현상 원천 차단
        x1 = int(cx - crop_w / 2)
        x1 = max(0, min(x1, frame_w - crop_w))
        x2 = x1 + crop_w

        # 5. 9:16 행렬 슬라이싱을 통한 물리적 크로핑 및 스트림 저장
        cropped_frame = frame[0:crop_h, x1:x2]
        out_video.write(cropped_frame)
        
        # 10프레임마다 터미널에 추적 상황 브로드캐스팅
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            print(f"⏳ Processing Frame: [{frame_idx}/{total_frames}] | Real-time Performance: {current_fps:.1f} FPS")

    # 6. 인프라 자원 해제
    cap.release()
    out_video.release()
    print("=" * 60)
    print(f"🎉 지능형 Auto Reframe 인코딩 완료! -> {OUTPUT_VIDEO_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    run_video_auto_reframe()