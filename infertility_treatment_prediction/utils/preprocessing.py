import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(train_path='./data/train.csv', test_path='./data/test.csv', submission_path='./data/sample_submission.csv'):
    """
    Load the dataset from the specified paths
    
    Args:
        train_path: Path to the training data
        test_path: Path to the test data
        submission_path: Path to the sample submission file
        
    Returns:
        train, test, sample_submission dataframes
    """
    train = pd.read_csv(train_path, encoding='utf-8')
    test = pd.read_csv(test_path, encoding='utf-8')
    sample_submission = pd.read_csv(submission_path, encoding='utf-8')
    return train, test, sample_submission

def handle_missing_values(train, test):
    """
    Handle missing values in both train and test datasets
    
    Args:
        train: Training dataframe
        test: Test dataframe
        
    Returns:
        train, test with handled missing values
    """
    # Numerical features with missing values
    num_features = [
        '임신 시도 또는 마지막 임신 경과 연수', '난자 채취 경과일', '난자 해동 경과일', 
        '난자 혼합 경과일', '배아 이식 경과일', '배아 해동 경과일'
    ]
    # Replace missing values with -1
    train[num_features] = train[num_features].fillna(-1)
    test[num_features] = test[num_features].fillna(-1)
    
    # Categorical features with missing values
    cat_features = ['착상 전 유전 검사 사용 여부', 'PGD 시술 여부', 'PGS 시술 여부']
    train[cat_features] = train[cat_features].fillna(-1)
    test[cat_features] = test[cat_features].fillna(-1)
    
    return train, test

def remove_outliers(train):
    """
    Remove outliers from training data
    
    Args:
        train: Training dataframe
        
    Returns:
        train with outliers removed
    """
    # Remove outliers in '배란 유도 유형'
    outlier_values = ['생식선 자극 호르몬', '세트로타이드 (억제제)']
    train = train[~train['배란 유도 유형'].isin(outlier_values)]
    
    return train

def preprocess_categorical(train, test):
    """
    Preprocess categorical variables by merging rare categories
    
    Args:
        train: Training dataframe
        test: Test dataframe
        
    Returns:
        train, test with preprocessed categorical variables
    """
    # Merge categories in '총 임신 횟수'
    merge_categories = ['3회', '4회', '5회', '6회 이상']
    train['총 임신 횟수'] = train['총 임신 횟수'].replace(merge_categories, '2회')
    test['총 임신 횟수'] = test['총 임신 횟수'].replace(merge_categories, '2회')
    
    # Rename '2회' to '2회 이상'
    train['총 임신 횟수'] = train['총 임신 횟수'].replace('2회', '2회 이상')
    test['총 임신 횟수'] = test['총 임신 횟수'].replace('2회', '2회 이상')
    
    # Merge categories in 'IVF 임신 횟수'
    merge_categories = ['3회', '4회', '5회', '6회 이상']
    train['IVF 임신 횟수'] = train['IVF 임신 횟수'].replace(merge_categories, '2회')
    test['IVF 임신 횟수'] = test['IVF 임신 횟수'].replace(merge_categories, '2회')
    
    # Rename '2회' to '2회 이상'
    train['IVF 임신 횟수'] = train['IVF 임신 횟수'].replace('2회', '2회 이상')
    test['IVF 임신 횟수'] = test['IVF 임신 횟수'].replace('2회', '2회 이상')
    
    # Merge categories in 'DI 임신 횟수'
    merge_categories = ['2회', '3회', '4회', '5회', '6회 이상']
    train['DI 임신 횟수'] = train['DI 임신 횟수'].replace(merge_categories, '1회')
    test['DI 임신 횟수'] = test['DI 임신 횟수'].replace(merge_categories, '1회')
    
    # Rename '1회' to '1회 이상'
    train['DI 임신 횟수'] = train['DI 임신 횟수'].replace('1회', '1회 이상')
    test['DI 임신 횟수'] = test['DI 임신 횟수'].replace('1회', '1회 이상')
    
    # Merge categories in '총 출산 횟수'
    merge_categories = ['3회', '4회', '5회', '6회 이상']
    train['총 출산 횟수'] = train['총 출산 횟수'].replace(merge_categories, '2회')
    test['총 출산 횟수'] = test['총 출산 횟수'].replace(merge_categories, '2회')
    
    # Rename '2회' to '2회 이상'
    train['총 출산 횟수'] = train['총 출산 횟수'].replace('2회', '2회 이상')
    test['총 출산 횟수'] = test['총 출산 횟수'].replace('2회', '2회 이상')
    
    # Merge categories in 'IVF 출산 횟수'
    merge_categories = ['3회', '4회', '5회', '6회 이상']
    train['IVF 출산 횟수'] = train['IVF 출산 횟수'].replace(merge_categories, '2회')
    test['IVF 출산 횟수'] = test['IVF 출산 횟수'].replace(merge_categories, '2회')
    
    # Rename '2회' to '2회 이상'
    train['IVF 출산 횟수'] = train['IVF 출산 횟수'].replace('2회', '2회 이상')
    test['IVF 출산 횟수'] = test['IVF 출산 횟수'].replace('2회', '2회 이상')
    
    # Merge categories in 'DI 출산 횟수'
    merge_categories = ['2회', '3회', '4회', '5회', '6회 이상']
    train['DI 출산 횟수'] = train['DI 출산 횟수'].replace(merge_categories, '1회')
    test['DI 출산 횟수'] = test['DI 출산 횟수'].replace(merge_categories, '1회')
    
    # Rename '1회' to '1회 이상'
    train['DI 출산 횟수'] = train['DI 출산 횟수'].replace('1회', '1회 이상')
    test['DI 출산 횟수'] = test['DI 출산 횟수'].replace('1회', '1회 이상')
    
    # Process '난자 기증자 나이'
    merge_ages = ['만20세 이하', '만21~25세']
    if '난자 기증자 나이' in train.columns:
        train['난자 기증자 나이'] = train['난자 기증자 나이'].replace(merge_ages, '만25세 이하')
        test['난자 기증자 나이'] = test['난자 기증자 나이'].replace(merge_ages, '만25세 이하')
    
    return train, test

def preprocess_numerical(train, test):
    """
    Convert numerical variables to categorical by binning
    
    Args:
        train: Training dataframe
        test: Test dataframe
        
    Returns:
        train, test with preprocessed numerical variables
    """
    # Process '이식된 배아 수'
    train['이식된 배아 수'] = pd.cut(train['이식된 배아 수'], 
                              bins=[0.0, 1.0, 2.0, 3.0, float('inf')], 
                              labels=['0개', '1개', '2개', '3개 이상'])
    test['이식된 배아 수'] = pd.cut(test['이식된 배아 수'], 
                             bins=[0.0, 1.0, 2.0, 3.0, float('inf')], 
                             labels=['0개', '1개', '2개', '3개 이상'])
    
    # Process '미세주입 배아 이식 수'
    train['미세주입 배아 이식 수'] = pd.cut(train['미세주입 배아 이식 수'], 
                                  bins=[0.0, 1.0, 2.0, 3.0, float('inf')], 
                                  labels=['0개', '1개', '2개', '3개 이상'])
    test['미세주입 배아 이식 수'] = pd.cut(test['미세주입 배아 이식 수'], 
                                 bins=[0.0, 1.0, 2.0, 3.0, float('inf')], 
                                 labels=['0개', '1개', '2개', '3개 이상'])
    
    # Process '난자 혼합 경과일'
    train['난자 혼합 경과일'] = pd.cut(train['난자 혼합 경과일'], 
                              bins=[-1.0, 0.0, 1.0, float('inf')], 
                              labels=['결측', '0일', '1일 이상'])
    test['난자 혼합 경과일'] = pd.cut(test['난자 혼합 경과일'], 
                             bins=[-1.0, 0.0, 1.0, float('inf')], 
                             labels=['결측', '0일', '1일 이상'])
    
    # Process '배아 이식 경과일'
    train['배아 이식 경과일'] = pd.cut(train['배아 이식 경과일'], 
                              bins=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float('inf')], 
                              labels=['결측', '0일', '1일', '2일', '3일', '4일', '5일', '6일 이상'])
    test['배아 이식 경과일'] = pd.cut(test['배아 이식 경과일'], 
                             bins=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float('inf')], 
                             labels=['결측', '0일', '1일', '2일', '3일', '4일', '5일', '6일 이상'])
    
    # Process '배아 해동 경과일'
    train['배아 해동 경과일'] = pd.cut(train['배아 해동 경과일'], 
                              bins=[-1.0, 0.0, 1.0, 2.0, 3.0, float('inf')], 
                              labels=['결측', '0일', '1일', '2일', '3일 이상'])
    test['배아 해동 경과일'] = pd.cut(test['배아 해동 경과일'], 
                             bins=[-1.0, 0.0, 1.0, 2.0, 3.0, float('inf')], 
                             labels=['결측', '0일', '1일', '2일', '3일 이상'])
    
    # Convert categorical object types to string type
    train['난자 혼합 경과일'] = train['난자 혼합 경과일'].astype(str)
    test['난자 혼합 경과일'] = test['난자 혼합 경과일'].astype(str)
    train['배아 이식 경과일'] = train['배아 이식 경과일'].astype(str)
    test['배아 이식 경과일'] = test['배아 이식 경과일'].astype(str)
    train['배아 해동 경과일'] = train['배아 해동 경과일'].astype(str)
    test['배아 해동 경과일'] = test['배아 해동 경과일'].astype(str)
    
    return train, test

def split_by_treatment_type(train, test):
    """
    Split data by treatment type (IVF or DI)
    
    Args:
        train: Training dataframe
        test: Test dataframe
        
    Returns:
        train_ivf, train_di, test_ivf, test_di
    """
    train_ivf = train[train['시술 유형'] == 'IVF']
    train_di = train[train['시술 유형'] == 'DI']
    test_ivf = test[test['시술 유형'] == 'IVF']
    test_di = test[test['시술 유형'] == 'DI']
    
    return train_ivf, train_di, test_ivf, test_di

def get_ivf_features():
    """
    Get feature list for IVF treatment
    
    Returns:
        list of features for IVF treatment
    """
    return [
        '시술 시기 코드', '시술 당시 나이', '배란 자극 여부', '배란 유도 유형', 
        '남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인', 
        '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인',
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', '불임 원인 - 자궁내막증',
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수',
        '착상 전 유전 진단 사용 여부', '총 생성 배아 수', '미세주입된 난자 수', '미세주입에서 생성된 배아 수', 
        '이식된 배아 수', '미세주입 배아 이식 수', '저장된 배아 수', '미세주입 후 저장된 배아 수', '해동된 배아 수', 
        '해동 난자 수', '수집된 신선 난자 수', '저장된 신선 난자 수', '혼합된 난자 수', 
        '파트너 정자와 혼합된 난자 수', '기증자 정자와 혼합된 난자 수',
        '동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부', '대리모 여부',
        '난자 혼합 경과일', '배아 이식 경과일', '배아 해동 경과일'
    ]

def get_di_features():
    """
    Get feature list for DI treatment
    
    Returns:
        list of features for DI treatment
    """
    return [
        '시술 시기 코드', '시술 당시 나이', '배란 자극 여부', '배란 유도 유형', 
        '남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', '여성 부 불임 원인', 
        '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인',
        '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', '불임 원인 - 자궁내막증',
        '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', 
        '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수',
        '난자 출처', '난자 기증자 나이', '정자 기증자 나이'
    ]

def encode_features(X, label_encoders=None, train=True):
    """
    Encode categorical features using LabelEncoder
    
    Args:
        X: Features dataframe
        label_encoders: Dictionary of existing label encoders (for test set)
        train: Boolean indicating if this is training data
        
    Returns:
        X with encoded features, label_encoders dictionary
    """
    if train:
        label_encoders = {}
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    else:
        for col in X.columns:
            if col in label_encoders:
                # Handle unseen categories by setting them to the most frequent class
                X[col] = X[col].astype(str)
                for cat in X[col].unique():
                    if cat not in label_encoders[col].classes_:
                        most_frequent = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
                        X.loc[X[col] == cat, col] = label_encoders[col].classes_[0]
                X[col] = label_encoders[col].transform(X[col])
    
    return X, label_encoders

def preprocess_data(train, test, process_ivf=True, process_di=True):
    """
    Complete preprocessing pipeline
    
    Args:
        train: Training dataframe
        test: Test dataframe
        process_ivf: Boolean indicating whether to process IVF data
        process_di: Boolean indicating whether to process DI data
        
    Returns:
        Dictionary with processed data and label encoders
    """
    # Handle missing values
    train, test = handle_missing_values(train, test)
    
    # Remove outliers from training data
    train = remove_outliers(train)
    
    # Preprocess categorical variables
    train, test = preprocess_categorical(train, test)
    
    # Preprocess numerical variables
    train, test = preprocess_numerical(train, test)
    
    # Split data by treatment type
    train_ivf, train_di, test_ivf, test_di = split_by_treatment_type(train, test)
    
    result = {
        'train': train,
        'test': test
    }
    
    # Process IVF data
    if process_ivf:
        ivf_features = get_ivf_features()
        
        # Extract feature columns that exist in the dataset
        available_ivf_features = [col for col in ivf_features if col in train_ivf.columns]
        
        # Prepare IVF data
        X_train_ivf = train_ivf[available_ivf_features].copy()
        y_train_ivf = train_ivf['임신 성공 여부']
        X_test_ivf = test_ivf[available_ivf_features].copy()
        
        # Encode features
        X_train_ivf, ivf_encoders = encode_features(X_train_ivf, train=True)
        X_test_ivf, _ = encode_features(X_test_ivf, ivf_encoders, train=False)
        
        result['X_train_ivf'] = X_train_ivf
        result['y_train_ivf'] = y_train_ivf
        result['X_test_ivf'] = X_test_ivf
        result['ivf_encoders'] = ivf_encoders
    
    # Process DI data
    if process_di:
        di_features = get_di_features()
        
        # Extract feature columns that exist in the dataset
        available_di_features = [col for col in di_features if col in train_di.columns]
        
        # Prepare DI data
        X_train_di = train_di[available_di_features].copy()
        y_train_di = train_di['임신 성공 여부']
        X_test_di = test_di[available_di_features].copy()
        
        # Encode features
        X_train_di, di_encoders = encode_features(X_train_di, train=True)
        X_test_di, _ = encode_features(X_test_di, di_encoders, train=False)
        
        result['X_train_di'] = X_train_di
        result['y_train_di'] = y_train_di
        result['X_test_di'] = X_test_di
        result['di_encoders'] = di_encoders
    
    return result 