import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

class BaseModel:
    """Base model class with common methods"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def train(self, X, y):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def predict_proba(self, X):
        """Get probability predictions"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        y_pred_proba = self.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Convert probabilities to binary predictions for accuracy
        y_pred = (y_pred_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }

class RandomForestModel(BaseModel):
    """RandomForest classifier for pregnancy success prediction"""
    
    def __init__(self, params=None):
        super().__init__()
        
        # Default parameters for RandomForest
        self.default_params = {
            'n_estimators': 500,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Use provided params or defaults
        self.params = params if params is not None else self.default_params
        self.model = RandomForestClassifier(**self.params)
        
    def train(self, X, y):
        """Train the RandomForest model"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        train_roc_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        val_roc_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        
        print(f"Training ROC AUC: {train_roc_auc:.4f}")
        print(f"Validation ROC AUC: {val_roc_auc:.4f}")
        
        return self.model
    
    def hyperparameter_search(self, X, y, param_grid=None, cv=5, n_iter=10):
        """Perform hyperparameter search with RandomizedSearchCV"""
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 10, 15, 20, 25],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best ROC AUC: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.params = random_search.best_params_
        self.model = RandomForestClassifier(**self.params)
        
        return random_search.best_params_

class XGBoostModel(BaseModel):
    """XGBoost classifier for pregnancy success prediction"""
    
    def __init__(self, params=None):
        super().__init__()
        
        # Default parameters for XGBoost based on previous tuning
        self.default_params = {
            'n_estimators': 325,
            'learning_rate': 0.139,
            'max_depth': 3,
            'subsample': 0.616,
            'colsample_bytree': 0.732,
            'gamma': 3.23,
            'min_child_weight': 9,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # Use provided params or defaults
        self.params = params if params is not None else self.default_params
        self.model = xgb.XGBClassifier(**self.params)
    
    def train(self, X, y):
        """Train the XGBoost model"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on validation set
        train_roc_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        val_roc_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        
        print(f"Training ROC AUC: {train_roc_auc:.4f}")
        print(f"Validation ROC AUC: {val_roc_auc:.4f}")
        
        return self.model
    
    def hyperparameter_search(self, X, y, param_grid=None, cv=3, n_iter=30):
        """Perform hyperparameter search with RandomizedSearchCV"""
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.7, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
                'gamma': [0, 1, 3, 5],
                'min_child_weight': [1, 3, 5, 7, 9]
            }
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best ROC AUC: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.params = random_search.best_params_
        self.model = xgb.XGBClassifier(**self.params)
        
        return random_search.best_params_

class ExtraTreesModel(BaseModel):
    """ExtraTrees classifier for pregnancy success prediction"""
    
    def __init__(self, params=None):
        super().__init__()
        
        # Default parameters for ExtraTrees
        self.default_params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Use provided params or defaults
        self.params = params if params is not None else self.default_params
        self.model = ExtraTreesClassifier(**self.params)
    
    def train(self, X, y):
        """Train the ExtraTrees model"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        train_roc_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        val_roc_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        
        print(f"Training ROC AUC: {train_roc_auc:.4f}")
        print(f"Validation ROC AUC: {val_roc_auc:.4f}")
        
        return self.model
    
    def hyperparameter_search(self, X, y, param_grid=None, cv=5, n_iter=10):
        """Perform hyperparameter search with RandomizedSearchCV"""
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 10, 15, 20, 25],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            estimator=ExtraTreesClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best ROC AUC: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.params = random_search.best_params_
        self.model = ExtraTreesClassifier(**self.params)
        
        return random_search.best_params_

class StackingEnsembleModel(BaseModel):
    """Stacking ensemble model combining multiple classifiers"""
    
    def __init__(self, estimators=None, final_estimator=None):
        super().__init__()
        
        # Default estimators if none provided
        if estimators is None:
            self.estimators = [
                ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=325, learning_rate=0.139, max_depth=3, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1))
            ]
        else:
            self.estimators = estimators
        
        # Default final estimator if none provided
        if final_estimator is None:
            self.final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.final_estimator = final_estimator
        
        # Initialize stacking classifier
        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def train(self, X, y):
        """Train the Stacking ensemble model"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        val_roc_auc = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        print(f"Validation ROC AUC: {val_roc_auc:.4f}")
        
        return self.model

def create_submission(ivf_preds, di_preds, test_ivf, test_di, sample_submission):
    """
    Create submission file from predictions
    
    Args:
        ivf_preds: Probability predictions for IVF samples
        di_preds: Probability predictions for DI samples
        test_ivf: Test IVF dataframe
        test_di: Test DI dataframe
        sample_submission: Sample submission dataframe
        
    Returns:
        submission dataframe
    """
    # Create a copy of the sample submission
    submission = sample_submission.copy()
    
    # Get test indices for IVF and DI
    ivf_indices = test_ivf.index
    di_indices = test_di.index
    
    # Populate submission with probabilities
    submission.loc[ivf_indices, '임신 성공 여부'] = ivf_preds
    submission.loc[di_indices, '임신 성공 여부'] = di_preds
    
    return submission 