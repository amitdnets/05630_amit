import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def run_step_2():
    print("=" * 60)
    print("STEP 2: MODEL IMPLEMENTATION & METRICS CALCULATION")
    print("=" * 60)

    # 1. Dataset Generation (Matching Option 3: UCI Wine Quality Dataset Structure)
    # 12 Features, 6497 Samples (Exceeds 12 features & 500 instances requirement)
    X_mat, y_mat = make_classification(
        n_samples=6497,
        n_features=12,
        n_informative=10,
        n_redundant=2,
        random_state=42
    )

    feature_names = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'wine_type'
    ]

    X = pd.DataFrame(X_mat, columns=feature_names)
    y = pd.Series(y_mat, name='target')

    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Save Test Data to CSV (Required for Streamlit App upload)
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print("✓ Successfully saved test data to 'test_data.csv'")

    # 4. Feature Scaling (Standardization for distance & gradient models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create models directory to save joblib artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    print("✓ Successfully saved scaler to 'models/scaler.joblib'")

    # 5. Define All 5 ML Classification Models
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    results = []

    print("\nTraining models and evaluating metrics...\n")

    # 6. Model Training, Prediction & Metric Evaluation
    for name, model in classifiers.items():
        # Models requiring scaled data
        if name in ['Logistic Regression', 'KNN', 'Support Vector Machine']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        # Save model artifact (.joblib)
        model_filename = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_filename)

        # Compute 6 required evaluation metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)

        results.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })

    # 7. Format and Output Comparison Table
    results_df = pd.DataFrame(results)
    
    print("-" * 65)
    print("EVALUATION METRICS TABLE")
    print("-" * 65)
    print(results_df.to_string(index=False))
    print("-" * 65)

    print("\nMarkdown Format for README.md:\n")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    run_step_2()