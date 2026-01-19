# Churn Prediction API

**Production-ready customer churn prediction system with FastAPI deployment.**

## 📋 Project Overview

This project predicts customer churn for a telecommunications company using machine learning. The system includes:
- End-to-end training pipeline
- REST API for real-time predictions
- 80% accuracy, 0.84 ROC-AUC

## 🏗️ Architecture
```
Data (CSV) → Preprocessing → Model Training → Saved Model → FastAPI → Predictions
```

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python3 data/download_data.py

# 3. Train model
python3 train_model.py

# 4. Run API
python3 -m uvicorn app.main:app --reload

# 5. Test
Open http://localhost:8000/docs
```

## 📁 Project Structure
```
churn-prediction-api/
├── data/
│   ├── download_data.py      # Downloads dataset
│   └── telco_churn.csv        # Customer data (7,043 rows)
├── models/
│   ├── model.pkl              # Trained RandomForest
│   ├── label_encoders.pkl     # Categorical encoders
│   ├── feature_names.pkl      # Feature list
│   └── metadata.pkl           # Performance metrics
├── app/
│   ├── main.py                # FastAPI application
│   ├── models.py              # Pydantic schemas (tomorrow)
│   └── ml_model.py            # ML inference logic (tomorrow)
├── train_model.py             # Training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔄 File Flow
```
1. download_data.py → Creates telco_churn.csv
2. train_model.py → Reads CSV, creates models/*.pkl
3. app/main.py → Loads models/*.pkl, serves API
4. Client → Calls API endpoints
```

## 📊 Model Performance

- **Accuracy:** 80.5%
- **ROC-AUC:** 0.847
- **Precision (Churn):** 65%
- **Recall (Churn):** 55%

## 🛠️ Built With

- **FastAPI** - Web framework
- **scikit-learn** - ML models
- **Pandas** - Data processing
- **Pydantic** - Data validation

## 📚 Learnings

This project applies concepts from *Designing Machine Learning Systems* by Chip Huyen:
- Separation of training and serving
- Model artifact versioning
- Health check endpoints
- Production considerations

## 🔜 Next Steps (Tuesday)

- [ ] Add POST /predict endpoint
- [ ] Implement Pydantic validation
- [ ] Add error handling
- [ ] Create batch prediction endpoint
- [ ] Add request logging

## 👤 Author

Lauren Williams - Microsoft Office of CTO Interview Project

## 📅 Timeline

- Monday: Model training + basic API
- Tuesday: Complete API + deployment considerations
```

---

## **SUMMARY: FILE RELATIONSHIPS**
```
EXECUTION ORDER:
1. download_data.py  →  Creates data/telco_churn.csv
2. train_model.py    →  Reads CSV, creates models/*.pkl
3. app/main.py       →  Loads *.pkl, serves HTTP API

DATA FLOW:
CSV → DataFrame → Preprocessed Features → Trained Model → Pickled Model → Loaded in API → Predictions

FILE DEPENDENCIES:
download_data.py:  No dependencies
train_model.py:    Depends on telco_churn.csv
app/main.py:       Depends on models/*.pkl

RUNTIME:
download_data.py:  Run once (or when data updates)
train_model.py:    Run when retraining needed
app/main.py:       Runs continuously as server