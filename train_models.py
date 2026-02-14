import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# 1. Load Data
df = pd.read_csv('heart.csv')

# 2. Preprocess Data
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split data - save the test set to upload to Streamlit later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_data = X_test.copy()
test_data['HeartDisease'] = y_test
test_data.to_csv('test_data.csv', index=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# 4. Train, Evaluate, and Save
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate metrics
    metrics = {
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    results.append(metrics)

    # Save the model artifact
    joblib.dump(model, f'model/{name.replace(" ", "_")}.pkl')

print(pd.DataFrame(results).to_markdown(index=False))