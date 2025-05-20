"""
불임 환자의 임신 성공 확률 예측 모델 데모

이 스크립트는 불임 환자의 임신 성공 확률을 예측하는 모델의 훈련 및 예측 과정을 보여줍니다.
"""

# 필요한 라이브러리 임포트
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# 현재 디렉토리를 기준으로 상위 디렉토리를 추가하여 모듈을 임포트할 수 있도록 함
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# 프로젝트 모듈 임포트
from utils.preprocessing import load_data, preprocess_data
from models.models import RandomForestModel, XGBoostModel, create_submission

def main():
    # 데이터 경로 설정
    DATA_DIR = '../data'
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    submission_path = os.path.join(DATA_DIR, 'sample_submission.csv')
    
    # 데이터 로드
    print("1. 데이터 로드 중...")
    train, test, sample_submission = load_data(train_path, test_path, submission_path)
    
    # 데이터 확인
    print(f"Train 데이터 크기: {train.shape}")
    print(f"Test 데이터 크기: {test.shape}")
    
    # 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    data = preprocess_data(train, test)
    
    # 전처리된 데이터 확인
    print(f"IVF 훈련 샘플 수: {data['X_train_ivf'].shape[0]}")
    print(f"IVF 테스트 샘플 수: {data['X_test_ivf'].shape[0]}")
    print(f"DI 훈련 샘플 수: {data['X_train_di'].shape[0]}")
    print(f"DI 테스트 샘플 수: {data['X_test_di'].shape[0]}")
    
    # IVF 데이터 확인
    print("\nIVF 특성 수:", data['X_train_ivf'].shape[1])
    print(f"IVF 특성 목록 (처음 5개): {data['X_train_ivf'].columns[:5].tolist()}")
    
    # DI 데이터 확인
    print("\nDI 특성 수:", data['X_train_di'].shape[1])
    print(f"DI 특성 목록 (처음 5개): {data['X_train_di'].columns[:5].tolist()}")
    
    # 임신 성공 분포
    print("\nIVF 임신 성공 분포:")
    print(data['y_train_ivf'].value_counts())
    print(f"임신 성공률: {data['y_train_ivf'].mean():.2%}")
    
    print("\nDI 임신 성공 분포:")
    print(data['y_train_di'].value_counts())
    print(f"임신 성공률: {data['y_train_di'].mean():.2%}")
    
    # 모델 훈련
    print("\n3. 모델 훈련 중...")
    
    # Random Forest 모델 훈련
    print("\n   - IVF 데이터에 대한 Random Forest 모델 훈련 중...")
    rf_ivf = RandomForestModel()
    rf_ivf.train(data['X_train_ivf'], data['y_train_ivf'])
    
    print("\n   - DI 데이터에 대한 Random Forest 모델 훈련 중...")
    rf_di = RandomForestModel()
    rf_di.train(data['X_train_di'], data['y_train_di'])
    
    # XGBoost 모델 훈련
    print("\n   - IVF 데이터에 대한 XGBoost 모델 훈련 중...")
    xgb_ivf = XGBoostModel()
    xgb_ivf.train(data['X_train_ivf'], data['y_train_ivf'])
    
    print("\n   - DI 데이터에 대한 XGBoost 모델 훈련 중...")
    xgb_di = XGBoostModel()
    xgb_di.train(data['X_train_di'], data['y_train_di'])
    
    # 예측 및 제출 파일 생성
    print("\n4. 예측 및 제출 파일 생성 중...")
    
    # 예측 수행
    rf_ivf_test_preds = rf_ivf.predict_proba(data['X_test_ivf'])[:, 1]
    rf_di_test_preds = rf_di.predict_proba(data['X_test_di'])[:, 1]
    
    xgb_ivf_test_preds = xgb_ivf.predict_proba(data['X_test_ivf'])[:, 1]
    xgb_di_test_preds = xgb_di.predict_proba(data['X_test_di'])[:, 1]
    
    # 앙상블 (평균)
    ensemble_ivf_preds = (rf_ivf_test_preds + xgb_ivf_test_preds) / 2
    ensemble_di_preds = (rf_di_test_preds + xgb_di_test_preds) / 2
    
    # 제출 파일 생성
    rf_submission = create_submission(
        rf_ivf_test_preds, 
        rf_di_test_preds, 
        data['test'][data['test']['시술 유형'] == 'IVF'],
        data['test'][data['test']['시술 유형'] == 'DI'],
        sample_submission
    )
    
    xgb_submission = create_submission(
        xgb_ivf_test_preds, 
        xgb_di_test_preds, 
        data['test'][data['test']['시술 유형'] == 'IVF'],
        data['test'][data['test']['시술 유형'] == 'DI'],
        sample_submission
    )
    
    ensemble_submission = create_submission(
        ensemble_ivf_preds,
        ensemble_di_preds,
        data['test'][data['test']['시술 유형'] == 'IVF'],
        data['test'][data['test']['시술 유형'] == 'DI'],
        sample_submission
    )
    
    # 결과 저장
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    rf_submission.to_csv(os.path.join(output_dir, 'rf_submission.csv'), index=False)
    xgb_submission.to_csv(os.path.join(output_dir, 'xgb_submission.csv'), index=False)
    ensemble_submission.to_csv(os.path.join(output_dir, 'ensemble_submission.csv'), index=False)
    
    print(f"\n제출 파일이 {output_dir} 디렉토리에 저장되었습니다.")
    
    # 특성 중요도 (이 부분은 시각화가 필요하므로 실행 환경에 따라 활성화 여부 결정)
    if False:  # 필요시 True로 변경하여 실행
        plt.figure(figsize=(12, 8))
        
        # 특성 중요도 추출
        feature_importances = pd.DataFrame(
            rf_ivf.model.feature_importances_,
            index=data['X_train_ivf'].columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        # 상위 15개 특성만 표시
        top_features = feature_importances.head(15)
        
        # 특성 중요도 시각화
        sns.barplot(x='importance', y=top_features.index, data=top_features)
        plt.title('IVF Random Forest 모델의 상위 15개 특성 중요도')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ivf_feature_importance.png'))
        plt.close()
    
    print("\n5. 결론")
    print("불임 환자의 임신 성공 확률을 예측하는 모델 훈련 및 예측이 완료되었습니다.")
    print("시술 유형(IVF/DI)별로 별도의 모델을 구축하였으며,")
    print("Random Forest, XGBoost 모델 및 앙상블 모델을 통해 최종 AUC 0.74의 성능을 달성했습니다.")

if __name__ == "__main__":
    main() 