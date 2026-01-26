"""
MODEL TRAINING PIPELINE
========================

Purpose:
    Complete end-to-end pipeline for training a churn prediction model.
    Handles data preprocessing, feature engineering, model training,
    evaluation, and artifact saving.

Input:
    - data/telco_churn.csv (raw customer data)

Processing Steps:
    1. Load data from CSV
    2. Drop customerID (not a feature)
    3. Handle TotalCharges data type issues
    4. Encode categorical variables (LabelEncoder)
    5. Split into train/test (80/20 stratified)
    6. Train RandomForestClassifier with class balancing
    7. Evaluate on test set
    8. Save model artifacts

Output Files (saved to models/):
    - model.pkl: Trained RandomForestClassifier
    - label_encoders.pkl: Dict of LabelEncoders for each categorical feature
    - feature_names.pkl: List of feature names in correct order
    - metadata.pkl: Dict with accuracy, AUC, feature list, model type

Model Hyperparameters:
    - n_estimators: 100 trees
    - max_depth: 10 (prevents overfitting)
    - class_weight: 'balanced' (handles 27% churn imbalance)
    - random_state: 42 (reproducibility)

Expected Performance:
    - Accuracy: ~80%
    - ROC-AUC: ~0.84
    - Recall (Churn): ~0.55 (catches 55% of churners)
    - Precision (Churn): ~0.65 (65% of churn predictions are correct)

Connection to Other Files:
    ← Reads: data/telco_churn.csv (from download_data.py)
    → Creates: models/*.pkl (used by app/main.py)
    
Why RandomForest?
    - Handles non-linear relationships
    - Robust to outliers
    - No feature scaling needed
    - Good accuracy vs speed trade-off
    - Built-in feature importance

When to Run:
    - Initially to create model
    - When retraining with new data
    - When tuning hyperparameters

Example Usage:
    $ python3 train_model.py
    🚀 Training Churn Prediction Model...
    ✅ Loaded 7043 customers
    ✅ Preprocessed 19 features
    🤖 Training Random Forest...
    📊 Results:
       Accuracy: 0.805
       ROC-AUC: 0.847
    ✅ Model saved to models/
"""

# Import and setup
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training Churn Prediction Model...")

# Load and explor data
# Data is a CSV but in production we would connect to a datawarehouse
df = pd.read_csv('data/telco_churn.csv')
print(f" Loaded {len(df)} customers")

# Quick preprocessing
"""
Transform total charges column from string to numeric
Fill in missing total charges with median. Median is robust to outliers
"""
df = df.drop('customerID', axis=1, errors='ignore')
if df['TotalCharges'].dtype == 'object':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Prepare Features and Target
# all features are included and target is the churn columns
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
"""
Find and encode each categorical columns, learn values and convert to numbers.
Save each encoder in label encoder dictionary, save for inference time later.
"""
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f" Preprocessed {len(X.columns)} features")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
print(" Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = model.score(X_test, y_test)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Results:")
print(f"   Accuracy: {accuracy:.3f}")
print(f"   ROC-AUC: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Save model artifacts
print("\n💾 Saving model...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

metadata = {
    'accuracy': accuracy,
    'roc_auc': auc,
    'features': X.columns.tolist(),
    'model_type': 'RandomForest'
}
with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Model saved to models/")
print("\n🎉 Training complete! Ready for API deployment.")