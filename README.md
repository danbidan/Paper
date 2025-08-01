# Paper - AI Research Environment

EEG Source Localization with Variational Autoencoder

## 프로젝트 구조

```
Paper/
├── main.py              # 메인 실행 파일
├── eeg_generator.py     # EEG 시뮬레이션 클래스
├── models.py           # PyTorch 모델 정의
├── data_generator.py   # 데이터 생성 및 처리
├── requirements.txt    # 필요한 패키지 목록
└── README.md          # 프로젝트 설명
```

## 설치 및 실행

### 1. Conda 환경 활성화
```bash
conda activate danbidan
```

### 2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
python main.py
```

## 주요 기능

### EEG 시뮬레이션 (`eeg_generator.py`)
- MNE-Python을 사용한 EEG 데이터 시뮬레이션
- 랜덤 dipole source 생성
- 노이즈 및 필터링 적용

### 딥러닝 모델 (`models.py`)
- **EEGEncoder**: EEG 신호를 잠재 공간으로 인코딩
- **EEGDecoder**: 잠재 공간에서 source activity로 디코딩
- Variational Autoencoder 구조

### 데이터 생성 (`data_generator.py`)
- 1000개 EEG 시뮬레이션 데이터 생성
- 랜덤 주파수(8-12Hz) 및 진폭 설정
- 예측 결과 분석 함수

## 사용 예시

```python
from data_generator import generate_eeg_dataset
from models import EEGEncoder, EEGDecoder

# 데이터 생성
all_eeg, all_labels = generate_eeg_dataset(n_trials=1000)

# 모델 초기화
encoder = EEGEncoder(input_dim=64, latent_dim=32)
decoder = EEGDecoder(latent_dim=32, output_dim=8196)

# 추론
eeg_tensor = torch.tensor(all_eeg[0])
mu, logvar = encoder(eeg_tensor)
z_samples = reparameterize(mu, logvar, n_samples=100)
preds = decoder(z_samples)
```

## 연구 목적

이 프로젝트는 EEG 신호로부터 뇌의 source activity를 추정하는 딥러닝 기반 source localization을 구현합니다. Variational Autoencoder를 사용하여 불확실성을 고려한 source 추정을 수행합니다.

힘든세상이에요
