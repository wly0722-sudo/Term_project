import os
from ultralytics import YOLO

DATASET_ROOT = os.path.abspath("./yolo_davis_dataset")
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")

def create_yaml():
    """train, val, test 세 가지 경로를 모두 명시한 세팅 파일 생성"""
    yaml_content = f"""
path: {DATASET_ROOT}
train: images/train
val: images/val
test: images/test

names:
  0: target_object
"""
    with open(YAML_PATH, 'w') as f:
        f.write(yaml_content.strip())
    print(f"📝 Test 경로가 추가된 {YAML_PATH} 생성 완료.")

def run_tuning_experiments():
    create_yaml()
    
    # [실험 1] SGD Tuning
    print("\n🏋️‍♂️ [실험 1] SGD Baseline Tuning 시작...")
    model1 = YOLO("yolov8n.pt")
    model1.train(data=YAML_PATH, epochs=10, imgsz=480, batch=16, optimizer='SGD', lr0=0.01, cos_lr=False, project="ml_project", name="yolo_davis_run1")

    # [실험 2] AdamW + Cosine Tuning
    print("\n🏋️‍♂️ [실험 2] AdamW Advanced Tuning 시작...")
    model2 = YOLO("yolov8n.pt")
    model2.train(data=YAML_PATH, epochs=10, imgsz=480, batch=16, optimizer='AdamW', lr0=0.001, cos_lr=True, project="ml_project", name="yolo_davis_run2")
    
    print("\n🎉 하이퍼파라미터 튜닝 완료! 이제 정석적인 파이프라인이 구축되었습니다.")

if __name__ == "__main__":
    run_tuning_experiments()