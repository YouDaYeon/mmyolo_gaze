# 👁️ Gaze Estimation Model

이 저장소는 이미지 내 인물의 시선을 예측하기 위한 **Gaze Estimation 모델**의 코드와 학습된 가중치(weight)를 제공합니다.  
본 모델은 head, face, eye 영역의 특징을 기반으로 3D 시선 벡터를 추정하도록 설계되었습니다.

---

## 📂 Repository Structure
```
├── mmyolo/ # Core MMYOLO package
│ ├── models/ # Model implementations (YOLO, Gaze models)
│ ├── datasets/ # Dataset and augmentation utilities
│ └── engine/ # Hook and Optimizer
├── configs/ # Configuration files for training/testing
├── tools/ # Scripts (train, test, conversion, analysis)
└── result/ # Experimental results (best weight (.pth))
```

## 🚀 Usage
### 1. 모델 다운로드
사전에 학습된 가중치(`.pth`) 파일을 `result/` 폴더에 저장한다.
### 2. 테스트 실행
```
python tools/test.py configs/yolov8/yolov8_s_fast_1xb12-40e_gaze_v0.py result/20250902_PositionMap_Eye/best_MAE_360_epoch_11.pth --show-dir test
```
테스트 결과 시각화 이미지가 'test/' 폴더에 저장된다.
### 3. 학습 실행
```
nohup python tools/train.py configs/yolov8/yolov8_s_fast_1xb12-40e_gaze_v0.py &> train_test.log &
```
nohup : 터미널 종료 후에도 학습 계속 진행 \
학습 로그는 'train_test.log' 파일에 저장된다.

## Docker

- 도커 빌드
    `docker build -t gaze_inference_final -f Dockerfile .`
    
- 컨테이너 실행
    `docker run -it --rm --gpus all -v /mnt/team_ai2:/mnt/team_ai2 gaze_inference_final bash`

## 📊 Results
| Dataset   | *Metric (Angular Error ↓)                | Description                    |
| --------- | ------------------------                | ------------------------------ |
| Gaze360   | 12.85° / 11.03° / 8.68° / 19.42°        | full Gaze360 dataset 이용       |
| Gaze360 subset | 10.13° / 9.92° / 7.81°             | Gaze360 中 detectable faces 이미지 이용 |

***Metric** : All 360° / Front 180° / Front-facing 20° / Backward

## 🌐 Dataset
[Gaze360 dataset download](https://gaze360.csail.mit.edu/)
