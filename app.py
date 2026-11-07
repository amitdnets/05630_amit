import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("📊 Machine Learning Classification Model Evaluator")
st.markdown("Upload test dataset (CSV) to evaluate pre-trained classification models without retraining.")

# Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Select Model for Evaluation",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/svm.joblib"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Test Data Preview")
    st.dataframe(df.head(10))

    if 'target' not in df.columns:
        st.error("Error: Uploaded dataset must include a 'target' column.")
    else:
        X_test = df.drop(columns=['target'])
        y_test = df['target']

        try:
            scaler = joblib.load("models/scaler.joblib")
            model = joblib.load(MODEL_FILES[model_choice])
        except Exception as e:
            st.error(f"Failed to load model artifact: {e}")
            st.stop()

        # Apply scaling to continuous features for scaled-dependent models
        if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_prob = y_pred

        # Compute all 6 required evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.markdown(f"### 🎯 Evaluation Metrics: **{model_choice}**")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC Score", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC Score", f"{mcc:.4f}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {model_choice}')
            st.pyplot(fig)

        with col2:
            st.subheader("📑 Detailed Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=4))
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar to run evaluations.")