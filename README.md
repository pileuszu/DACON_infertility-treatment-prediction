# Prediction of Pregnancy Success in Infertility Patients

This project develops a machine learning model to predict pregnancy success rates in infertility patients. The model achieves an accuracy of 0.74 by building separate prediction models for IVF (In Vitro Fertilization) and DI (Donor Insemination) treatments, utilizing various machine learning algorithms and ensemble techniques.

## Project Overview

The success of pregnancy in infertility patients is determined by multiple factors. This project develops a model that predicts pregnancy success probability using various features including treatment information, medical characteristics, and patient history. The data is divided into IVF and DI treatments, with optimized models built for each type.

## Data

The dataset consists of the following files:
- `train.csv`: Training data (patient characteristics and pregnancy success)
- `test.csv`: Test data (patient characteristics)
- `sample_submission.csv`: Submission format file
- `데이터 명세.xlsx`: Data specification file

Key Features:
- Treatment type (IVF/DI)
- Patient age
- Ovulation-related information
- Infertility causes
- Previous treatment and pregnancy/birth history
- Egg/embryo-related information (mainly for IVF)

## File Structure

```
infertility_treatment_prediction/
├── data/                      # Data files directory
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                    # Model class definitions
│   ├── __init__.py
│   └── models.py              # Model classes (RandomForest, XGBoost, ExtraTrees, Stacking)
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing functions
│   └── copy_data.py          # Data copying utility
├── notebooks/                 # Analysis notebooks
├── __init__.py
└── train.py                   # Model training and prediction script
```

## Methodology

### Data Preprocessing

- Missing value handling: Replace missing values with -1 for both numerical and categorical variables
- Outlier removal: Remove rare values in 'ovulation induction type'
- Category merging: Merge rare categories to improve model performance
- Numerical variable binning: Convert certain numerical variables to categorical
- Treatment type separation: Separate IVF and DI data for individual model training

### Modeling

The following models were trained separately for each treatment type (IVF/DI):

1. **Random Forest**: Ensemble of decision trees to prevent overfitting
2. **XGBoost**: Gradient boosting-based model for high prediction performance
3. **Extra Trees**: Similar to Random Forest but with increased randomness
4. **Stacking Ensemble**: Combines predictions from multiple models for improved performance

### Evaluation Metrics

ROC AUC (Receiver Operating Characteristic Area Under Curve) was used as the main evaluation metric. The final model achieved a ROC AUC score of 0.74.

## Results

- Random Forest: ROC AUC 0.72
- XGBoost: ROC AUC 0.73
- Extra Trees: ROC AUC 0.71
- Stacking Ensemble: ROC AUC 0.74

## Usage

### Environment Setup

```bash
# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Data Preparation

Copy original data to project structure:

```bash
python -m infertility_treatment_prediction.utils.copy_data --source_dir PATH_TO_ORIGINAL_DATA --target_dir infertility_treatment_prediction/data
```

### Model Training and Prediction

```bash
# Train all models and create ensemble predictions
python -m infertility_treatment_prediction.train --models all

# Train specific models
python -m infertility_treatment_prediction.train --models rf xgb
```

### Options

- `--train_path`: Training data path
- `--test_path`: Test data path
- `--submission_path`: Sample submission file path
- `--output_dir`: Results save directory
- `--models`: Models to train (rf, xgb, et, stacking, all)

## License

This project is distributed under the MIT License. 