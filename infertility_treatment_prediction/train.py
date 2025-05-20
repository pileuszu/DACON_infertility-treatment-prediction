import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import load_data, preprocess_data
from models.models import (
    RandomForestModel, 
    XGBoostModel, 
    ExtraTreesModel, 
    StackingEnsembleModel,
    create_submission
)

def main(args):
    print("Starting pregnancy success prediction training...")
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train, test, sample_submission = load_data(
        args.train_path,
        args.test_path,
        args.submission_path
    )
    
    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(train, test)
    
    # Access preprocessed data
    X_train_ivf = data['X_train_ivf']
    y_train_ivf = data['y_train_ivf']
    X_test_ivf = data['X_test_ivf']
    
    X_train_di = data['X_train_di']
    y_train_di = data['y_train_di']
    X_test_di = data['X_test_di']
    
    print(f"IVF training samples: {X_train_ivf.shape[0]}")
    print(f"IVF test samples: {X_test_ivf.shape[0]}")
    print(f"DI training samples: {X_train_di.shape[0]}")
    print(f"DI test samples: {X_test_di.shape[0]}")
    
    # Initialize predictions
    ivf_preds = np.zeros(X_test_ivf.shape[0])
    di_preds = np.zeros(X_test_di.shape[0])
    
    # Train and predict with Random Forest for IVF
    if 'rf' in args.models or 'all' in args.models:
        print("\n=== Training Random Forest for IVF data ===")
        rf_ivf = RandomForestModel()
        rf_ivf.train(X_train_ivf, y_train_ivf)
        
        # Get predictions for IVF data
        ivf_rf_preds = rf_ivf.predict_proba(X_test_ivf)[:, 1]
        
        print("\n=== Training Random Forest for DI data ===")
        rf_di = RandomForestModel()
        rf_di.train(X_train_di, y_train_di)
        
        # Get predictions for DI data
        di_rf_preds = rf_di.predict_proba(X_test_di)[:, 1]
        
        # Create Random Forest submission
        rf_submission = create_submission(
            ivf_rf_preds, 
            di_rf_preds, 
            data['test'][data['test']['시술 유형'] == 'IVF'],
            data['test'][data['test']['시술 유형'] == 'DI'],
            sample_submission
        )
        rf_submission.to_csv(os.path.join(args.output_dir, 'rf_submission.csv'), index=False)
        
        # Add to ensemble predictions
        ivf_preds += ivf_rf_preds
        di_preds += di_rf_preds
    
    # Train and predict with XGBoost for IVF
    if 'xgb' in args.models or 'all' in args.models:
        print("\n=== Training XGBoost for IVF data ===")
        xgb_ivf = XGBoostModel()
        xgb_ivf.train(X_train_ivf, y_train_ivf)
        
        # Get predictions for IVF data
        ivf_xgb_preds = xgb_ivf.predict_proba(X_test_ivf)[:, 1]
        
        print("\n=== Training XGBoost for DI data ===")
        xgb_di = XGBoostModel()
        xgb_di.train(X_train_di, y_train_di)
        
        # Get predictions for DI data
        di_xgb_preds = xgb_di.predict_proba(X_test_di)[:, 1]
        
        # Create XGBoost submission
        xgb_submission = create_submission(
            ivf_xgb_preds, 
            di_xgb_preds, 
            data['test'][data['test']['시술 유형'] == 'IVF'],
            data['test'][data['test']['시술 유형'] == 'DI'],
            sample_submission
        )
        xgb_submission.to_csv(os.path.join(args.output_dir, 'xgb_submission.csv'), index=False)
        
        # Add to ensemble predictions
        ivf_preds += ivf_xgb_preds
        di_preds += di_xgb_preds
    
    # Train and predict with ExtraTrees for IVF
    if 'et' in args.models or 'all' in args.models:
        print("\n=== Training ExtraTrees for IVF data ===")
        et_ivf = ExtraTreesModel()
        et_ivf.train(X_train_ivf, y_train_ivf)
        
        # Get predictions for IVF data
        ivf_et_preds = et_ivf.predict_proba(X_test_ivf)[:, 1]
        
        print("\n=== Training ExtraTrees for DI data ===")
        et_di = ExtraTreesModel()
        et_di.train(X_train_di, y_train_di)
        
        # Get predictions for DI data
        di_et_preds = et_di.predict_proba(X_test_di)[:, 1]
        
        # Create ExtraTrees submission
        et_submission = create_submission(
            ivf_et_preds, 
            di_et_preds, 
            data['test'][data['test']['시술 유형'] == 'IVF'],
            data['test'][data['test']['시술 유형'] == 'DI'],
            sample_submission
        )
        et_submission.to_csv(os.path.join(args.output_dir, 'et_submission.csv'), index=False)
        
        # Add to ensemble predictions
        ivf_preds += ivf_et_preds
        di_preds += di_et_preds
    
    # Train and predict with Stacking Ensemble
    if 'stacking' in args.models or 'all' in args.models:
        print("\n=== Training Stacking Ensemble for IVF data ===")
        stacking_ivf = StackingEnsembleModel()
        stacking_ivf.train(X_train_ivf, y_train_ivf)
        
        # Get predictions for IVF data
        ivf_stacking_preds = stacking_ivf.predict_proba(X_test_ivf)[:, 1]
        
        print("\n=== Training Stacking Ensemble for DI data ===")
        stacking_di = StackingEnsembleModel()
        stacking_di.train(X_train_di, y_train_di)
        
        # Get predictions for DI data
        di_stacking_preds = stacking_di.predict_proba(X_test_di)[:, 1]
        
        # Create Stacking submission
        stacking_submission = create_submission(
            ivf_stacking_preds, 
            di_stacking_preds, 
            data['test'][data['test']['시술 유형'] == 'IVF'],
            data['test'][data['test']['시술 유형'] == 'DI'],
            sample_submission
        )
        stacking_submission.to_csv(os.path.join(args.output_dir, 'stacking_submission.csv'), index=False)
        
        # Replace ensemble predictions with stacking (better approach)
        ivf_preds = ivf_stacking_preds
        di_preds = di_stacking_preds
    
    # If multiple models selected but not stacking, average the predictions
    if len([m for m in args.models if m != 'stacking' and m != 'all']) > 1 and 'stacking' not in args.models:
        num_models = sum(['rf' in args.models or 'all' in args.models, 
                           'xgb' in args.models or 'all' in args.models,
                           'et' in args.models or 'all' in args.models])
        ivf_preds /= num_models
        di_preds /= num_models
    
    # Create final submission
    final_submission = create_submission(
        ivf_preds, 
        di_preds, 
        data['test'][data['test']['시술 유형'] == 'IVF'],
        data['test'][data['test']['시술 유형'] == 'DI'],
        sample_submission
    )
    final_submission.to_csv(os.path.join(args.output_dir, 'final_submission.csv'), index=False)
    
    print(f"\nTraining completed! Submission files saved to {args.output_dir}")
    print("Submission files: ")
    for model in args.models:
        if model == 'all':
            print("  - rf_submission.csv")
            print("  - xgb_submission.csv")
            print("  - et_submission.csv")
            if 'stacking' in args.models:
                print("  - stacking_submission.csv")
            break
        elif model == 'rf':
            print("  - rf_submission.csv")
        elif model == 'xgb':
            print("  - xgb_submission.csv")
        elif model == 'et':
            print("  - et_submission.csv")
        elif model == 'stacking':
            print("  - stacking_submission.csv")
    print("  - final_submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pregnancy Success Prediction Training")
    
    parser.add_argument('--train_path', type=str, default='./data/train.csv',
                       help='Path to training data')
    parser.add_argument('--test_path', type=str, default='./data/test.csv',
                       help='Path to test data')
    parser.add_argument('--submission_path', type=str, default='./data/sample_submission.csv',
                       help='Path to sample submission file')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for saving models and submissions')
    parser.add_argument('--models', nargs='+', default=['all'],
                       choices=['rf', 'xgb', 'et', 'stacking', 'all'],
                       help='Models to train: rf (Random Forest), xgb (XGBoost), et (ExtraTrees), stacking (Stacking Ensemble), all')
    
    args = parser.parse_args()
    main(args) 