"""
Churn Prediction Model Training Pipeline
Trains a model to predict customer churn
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CHURN PREDICTION MODEL TRAINING")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
df = pd.read_csv('data/telco_churn.csv')
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing data...")

# Drop customer ID (not a feature)
df = df.drop('customerID', axis=1, errors='ignore')

# Handle TotalCharges (sometimes stored as string)
if df['TotalCharges'].dtype == 'object':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols[:5]}...")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("✅ Encoded categorical variables")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled (mean=0, std=1)")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n🤖 Training models...")

# Model 1: Logistic Regression (fast, interpretable)
print("\n1️⃣  Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)

print(f"   Accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")
print(f"   ROC-AUC: {lr_auc:.3f}")

# Model 2: Random Forest (better accuracy)
print("\n2️⃣  Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # No scaling needed for RF

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"   Accuracy: {rf_model.score(X_test, y_test):.3f}")
print(f"   ROC-AUC: {rf_auc:.3f}")

# Choose best model
print("\n🏆 Model Selection:")
if rf_auc > lr_auc:
    print(f"   ✅ Random Forest selected (AUC: {rf_auc:.3f})")
    best_model = rf_model
    best_name = "RandomForest"
    use_scaling = False
else:
    print(f"   ✅ Logistic Regression selected (AUC: {lr_auc:.3f})")
    best_model = lr_model
    best_name = "LogisticRegression"
    use_scaling = True

# ============================================================================
# 6. EVALUATION
# ============================================================================
print("\n📈 Detailed Evaluation:")
if use_scaling:
    final_pred = lr_model.predict(X_test_scaled)
    final_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
else:
    final_pred = rf_model.predict(X_test)
    final_proba = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, final_pred, target_names=['No Churn', 'Churn']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, final_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# ============================================================================
# 7. FEATURE IMPORTANCE (if Random Forest)
# ============================================================================
if best_name == "RandomForest":
    print("\n📊 Top 10 Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# ============================================================================
# 8. SAVE MODEL ARTIFACTS
# ============================================================================
print("\n💾 Saving model artifacts...")

# Save model
model_path = f'models/{best_name.lower()}_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✅ Model saved: {model_path}")

# Save scaler (if used)
if use_scaling:
    scaler_path = 'models/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {scaler_path}")

# Save label encoders
encoders_path = 'models/label_encoders.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Label encoders saved: {encoders_path}")

# Save feature names
feature_names_path = 'models/feature_names.pkl'
with open(feature_names_path, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print(f"✅ Feature names saved: {feature_names_path}")

# Save metadata
metadata = {
    'model_type': best_name,
    'use_scaling': use_scaling,
    'features': X.columns.tolist(),
    'categorical_features': categorical_cols,
    'numerical_features': numerical_cols,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': best_model.score(X_test_scaled if use_scaling else X_test, y_test),
    'roc_auc': rf_auc if best_name == "RandomForest" else lr_auc,
    'churn_rate': (df['Churn'] == 'Yes').mean()
}

metadata_path = 'models/metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✅ Metadata saved: {metadata_path}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\nModel: {best_name}")
print(f"Accuracy: {metadata['accuracy']:.3f}")
print(f"ROC-AUC: {metadata['roc_auc']:.3f}")
print(f"\nNext step: Build FastAPI wrapper in app/main.py")