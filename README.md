# Machine Learning Assignment 2 - Model Evaluation & Deployment

## a. Problem Statement
The objective of this assignment is to evaluate five machine learning classification models on a structured tabular dataset (12 features, 6,497 samples) and deploy an interactive Streamlit application to evaluate models on test datasets without retraining.

## b. Dataset Description
- **Dataset Name**: UCI Wine Quality Dataset (Combined Red & White)
- **Source**: UCI Machine Learning Repository / Kaggle
- **Problem Type**: Binary Classification
- **Total Instances**: 6,497
- **Total Features**: 12 Numerical Input Features (`fixed_acidity`, `volatile_acidity`, `citric_acid`, `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `wine_type`)
- **Target Variable**: Binary Target (`0` or `1`)
- **Train/Test Split**: 80% Train / 20% Test (1,300 test instances exported to `test_data.csv`)

## c. Github Repository Link
https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>

## d. Models Used

### Performance Comparison Table

| **ML Model Name**          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:---------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| **Logistic Regression**    |     0.8562 | 0.9281 |      0.8502 |   0.8646 | 0.8574 | 0.7124 |
| **Decision Tree**          |     0.8538 | 0.8538 |      0.856  |   0.8508 | 0.8534 | 0.7077 |
| **KNN**                    |     0.9662 | 0.9886 |      0.9825 |   0.9492 | 0.9656 | 0.9328 |
| **Naive Bayes**            |     0.8038 | 0.8869 |      0.817  |   0.7831 | 0.7997 | 0.6082 |
| **Support Vector Machine** |     0.9685 | 0.991  |      0.978  |   0.9585 | 0.9681 | 0.9371 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundaries yield a strong performance baseline (AUC: 0.9281) with balanced precision and recall. |
| **Decision Tree** | Captures feature interactions well, but lower AUC (0.8538) shows step-function boundary limitations on continuous features. |
| **KNN** | Achieves outstanding results (Accuracy: 96.62%, MCC: 0.9328) because feature scaling ensures effective distance computations. |
| **Naive Bayes** | Performs decently as a fast baseline model, but slight feature correlations reduce overall MCC (0.6082). |
| **Support Vector Machine** | RBF kernel maps continuous feature relations effectively, delivering peak accuracy (96.85%) and highest AUC (0.9910). |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the top scores across Accuracy, AUC, and MCC on this dataset. |