import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart Disease Classification Web App")

# Feature a: Dataset upload option
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader("Upload your test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
   
    if 'HeartDisease' in data.columns:
        X_test = data.drop('HeartDisease', axis=1)
        y_test = data['HeartDisease']
       
        scaler = joblib.load('model/scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
       
        # Feature b: Model selection dropdown
        st.header("2. Select Model")
        model_options = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost']
        selected_model_name = st.selectbox("Choose a classification model:", model_options)
       
        if st.button("Run Evaluation"):
            model_path = f'model/{selected_model_name.replace(" ", "_")}.pkl'
            model = joblib.load(model_path)
           
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
           
            # Feature c: Display of evaluation metrics
            st.header("3. Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
            col2.metric("AUC", round(roc_auc_score(y_test, y_proba), 4))
            col3.metric("Precision", round(precision_score(y_test, y_pred), 4))
           
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
            col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
            col6.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))
           
            # Feature d: Confusion matrix
            st.header("4. Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
    else:
        st.error("Uploaded CSV must contain the 'HeartDisease' target column.") 