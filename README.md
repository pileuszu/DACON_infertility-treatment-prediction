# 불임 환자의 임신 성공 확률 예측 (Prediction of Pregnancy Success in Infertility Patients)

이 프로젝트는 불임 환자의 임신 성공 확률을 예측하는 머신러닝 모델을 개발합니다. 시술 유형에 따라 IVF(체외수정)와 DI(인공수정)로 나누어 예측 모델을 구축하였으며, 여러 머신러닝 알고리즘과 앙상블 기법을 활용하여 예측 정확도 0.74를 달성하였습니다.

## 프로젝트 개요

불임 환자의 임신 성공 여부는 여러 요인에 의해 결정됩니다. 이 프로젝트에서는 환자의 시술 정보, 의학적 특성, 과거 병력 등 다양한 특성을 활용하여 임신 성공 확률을 예측하는 모델을 개발하였습니다. 데이터는 시술 유형에 따라 IVF(체외수정)와 DI(인공수정)로 구분되며, 각 유형별로 최적화된 모델을 구축하였습니다.

## 데이터

데이터는 다음 파일로 구성되어 있습니다:
- `train.csv`: 훈련 데이터 (환자 특성 및 임신 성공 여부)
- `test.csv`: 테스트 데이터 (환자 특성)
- `sample_submission.csv`: 제출 양식 파일
- `데이터 명세.xlsx`: 데이터 설명 파일

주요 특성:
- 시술 유형 (IVF/DI)
- 환자 나이
- 배란 관련 정보
- 불임 원인
- 과거 시술 및 임신/출산 이력
- 난자/배아 관련 정보 (주로 IVF에 해당)

## 파일 구조

```
infertility_treatment_prediction/
├── data/                      # 데이터 파일 디렉토리
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                    # 모델 클래스 정의
│   ├── __init__.py
│   └── models.py              # 모델 클래스 (RandomForest, XGBoost, ExtraTrees, Stacking)
├── utils/                     # 유틸리티 함수
│   ├── __init__.py
│   ├── preprocessing.py       # 데이터 전처리 함수
│   └── copy_data.py           # 데이터 복사 유틸리티
├── notebooks/                 # 분석 노트북
├── __init__.py
└── train.py                   # 모델 훈련 및 예측 스크립트
```

## 방법론

### 데이터 전처리

- 결측값 처리: 수치형 및 범주형 변수의 결측값을 -1로 대체
- 이상치 제거: '배란 유도 유형'에서 드문 값 제거
- 범주 병합: 발생 빈도가 낮은 범주를 병합하여 모델 성능 향상
- 수치형 변수 범주화: 일부 수치형 변수를 범주형으로 변환
- 시술 유형별 분리: IVF와 DI 데이터를 분리하여 별도의 모델 학습

### 모델링

다음 모델들을 사용하여 각 시술 유형(IVF/DI)별로 별도의 모델을 학습시켰습니다:

1. **Random Forest**: 다양한 결정 트리를 앙상블하여 과적합을 방지
2. **XGBoost**: 그래디언트 부스팅 기반으로 높은 예측 성능 제공
3. **Extra Trees**: Random Forest와 유사하나 무작위성을 더 활용
4. **Stacking Ensemble**: 여러 모델의 예측을 결합하여 성능 향상

### 평가 지표

ROC AUC(Receiver Operating Characteristic Area Under Curve)를 주요 평가 지표로 사용하였습니다. 최종 모델은 0.74의 ROC AUC 점수를 달성하였습니다.

## 결과

- Random Forest: ROC AUC 0.72
- XGBoost: ROC AUC 0.73
- Extra Trees: ROC AUC 0.71
- Stacking Ensemble: ROC AUC 0.74

## 사용 방법

### 환경 설정

```bash
# 가상 환경 생성 및 활성화 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요 패키지 설치
pip install -r requirements.txt
```

### 데이터 준비

원본 데이터를 프로젝트 구조에 맞게 복사:

```bash
python -m infertility_treatment_prediction.utils.copy_data --source_dir PATH_TO_ORIGINAL_DATA --target_dir infertility_treatment_prediction/data
```

### 모델 훈련 및 예측

```bash
# 모든 모델 학습 및 앙상블 예측
python -m infertility_treatment_prediction.train --models all

# 특정 모델만 학습
python -m infertility_treatment_prediction.train --models rf xgb
```

### 옵션

- `--train_path`: 훈련 데이터 경로
- `--test_path`: 테스트 데이터 경로
- `--submission_path`: 샘플 제출 파일 경로
- `--output_dir`: 결과물 저장 디렉토리
- `--models`: 학습할 모델 선택 (rf, xgb, et, stacking, all)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 참고 자료

- 시술 유형별 데이터 분리의 중요성에 대한 선행 연구
- 범주 병합을 통한 모델 성능 향상에 관한 논문
- 앙상블 기법을 통한 불임 환자 임신 성공 예측 연구 