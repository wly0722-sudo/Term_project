# 🎯 YOLOv8 기반 객체 추적 및 지능형 오토 리프레임(Auto-Reframe) 파이프라인

이 프로젝트는 머신러닝/딥러닝 텀 프로젝트로, **DAVIS 2016 데이터셋**을 활용하여 YOLOv8 모델을 파인튜닝하고 동영상 내 타겟 객체를 실시간으로 추적하는 시스템을 구축합니다. 
나아가 추적된 객체의 위치 정보를 기반으로 **16:9 가로형 비디오를 9:16 세로형 숏폼(Shorts) 비디오로 자동 크롭 및 변환(Auto-Reframe)**하는 파이프라인을 제안합니다.

<br>

## 💡 주요 특징 및 실험 내용

1. **시퀀스 레벨 데이터 격리 (Sequence-level Split)**
   - 데이터 누수(Data Leakage)를 방지하기 위해 프레임 단위가 아닌 비디오 폴더(시퀀스) 단위로 Train/Val/Test 셋을 분할(70:15:15)하여 커스텀 YOLO 데이터셋을 구축합니다.
2. **옵티마이저 하이퍼파라미터 튜닝 (Optimizer Tuning)**
   - `Run1 (SGD)` vs `Run2 (AdamW + Cosine Annealing)` 대조 학습을 통해 객체 탐지 모델의 최적의 가중치를 탐색합니다.
3. **전통적 CV 추적기(CSRT)와의 대조군 시뮬레이션**
   - 전통적인 OpenCV의 CSRT 추적기법의 한계점(빠른 모션, 가림 현상 등)을 시각화하고 딥러닝 기반 YOLOv8 ByteTrack 추적 기법과 IoU 및 FPS 성능을 비교합니다.
4. **인퍼런스 하이퍼파라미터 소거 연구 (Ablation Study)**
   - 추적 파라미터인 `Confidence Threshold`와 `NMS IoU`의 임계값 조절(Strict vs Flexible 세팅)이 객체 추적 유지력에 미치는 영향을 실험합니다.
5. **지능형 오토 리프레임 파이프라인 (Auto-Reframe Application)**
   - 파인튜닝된 YOLOv8 추적기를 사용하여 동영상 속 메인 객체가 항상 프레임 중앙에 위치하도록 9:16 비율로 자동 자르기(Cropping)를 수행합니다.

<br>

## 📁 파일 구성 요소

### 1. 데이터셋 및 학습
- `make_yolo_dataset.py`: DAVIS 2016 데이터를 YOLOv8 포맷으로 자동 변환하고 시퀀스 단위로 Train/Val/Test 폴더를 분할합니다.
- `train_yolo.py`: 구성된 데이터셋을 바탕으로 YOLOv8n 모델을 SGD 및 AdamW 설정으로 각각 학습시킵니다.

### 2. 시각화 및 성능 평가
- `baseline_visuallize.py` / `opencv_fail_visualize.py`: 기존 OpenCV 기반 CSRT 추적 알고리즘의 동작 및 한계를 시각화합니다.
- `yolov8_visualize.py`: 사전 학습된 베이스라인 YOLOv8 모델의 추적 성능을 렌더링합니다.
- `yolov8_trained_visualize.py`: 파인튜닝이 완료된 모델(Run1, Run2)들의 성능을 평가하고 비디오에 GT, BBox, IoU, FPS 메트릭을 오버레이하여 저장합니다.

### 3. 소거 연구 (Ablation Study)
- `Iou_cof_run1.py` / `Iou_cof_run2.py`: 각각의 학습된 모델(Run1, Run2)을 대상으로 Strict(Conf 0.5/IoU 0.3)와 Flexible(Conf 0.25/IoU 0.6) 조건에서의 변화를 탐구합니다.
- `hyper_parameter.py`: 최상의 모델 성적 도출을 위해 튜닝 세팅별 대조 시각화를 총괄합니다.

### 4. 핵심 응용 파이프라인
- `video_reframe_pipeline_yolo_ver.py`: 최적화가 끝난 가중치와 하이퍼파라미터를 최종 적용하여, 임의의 16:9 와이드 영상을 9:16 비율의 숏폼으로 변환하는 자동화 엔지니어링 스크립트입니다.

<br>

## 🚀 실행 가이드 (How to Use)

### Step 0. 요구 사항 (Requirements)
```bash
pip install opencv-python ultralytics numpy
```
*본 프로젝트는 루트 디렉토리에 `DAVIS` (DAVIS 2016 데이터셋 폴더)가 존재해야 작동합니다.*

### Step 1. YOLO 포맷 데이터셋 생성
```bash
python make_yolo_dataset.py
```
실행 후 `yolo_davis_dataset` 폴더가 생성되며 시퀀스 레벨로 분할된 데이터가 준비됩니다.

### Step 2. 모델 파인튜닝
```bash
python train_yolo.py
```
실행이 완료되면 `./runs/detect/ml_project/` 디렉토리에 `yolo_davis_run1`(SGD)과 `yolo_davis_run2`(AdamW)의 가중치가 저장됩니다.

### Step 3. 하이퍼파라미터 및 대조 시각화 검증
학습된 가중치를 평가하고, 파라미터 소거 연구 비디오를 생성합니다. (원하는 스크립트 실행)
```bash
python yolov8_trained_visualize.py
python hyper_parameter.py
python opencv_fail_visualize.py
```

### Step 4. 지능형 오토 리프레임 (Auto-Reframe) 파이프라인 가동
16:9의 원본 비디오를 `input_video.mp4` 이름으로 프로젝트 루트에 준비한 후 아래 명령을 실행하세요.
```bash
python video_reframe_pipeline_yolo_ver.py
```
성공적으로 실행되면 세로형 비디오(`output_reframe_9_16.mp4`)가 생성됩니다.

<br>

## 📊 평가 지표 (Metrics)
모든 검증 및 테스트 비디오는 프레임 좌측 상단에 실시간 성능 스코어보드를 제공합니다.
- **IoU (Intersection over Union)**: 예측된 바운딩 박스와 정답(Ground Truth) 마스크 박스 간의 겹침 비율. 정확성을 대변합니다.
- **FPS (Frames Per Second)**: 실시간 추적 속도 연산. 알고리즘의 가벼움과 실효성을 증명합니다.

<br>

## 🛡️ License 및 Acknowledgements
- Dataset: DAVIS (Densely-Annotated Video Segmentation) 2016
- Model Framework: Ultralytics YOLOv8