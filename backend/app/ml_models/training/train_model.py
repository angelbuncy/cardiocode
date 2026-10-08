"""
CardioCode – ML Training Script
════════════════════════════════════════════════════════════════════

DATA SOURCE (Open & Free):
────────────────────────────────────────────────────────────────────
  UCI Heart Disease Dataset (Cleveland subset)
  URL  : https://archive.ics.uci.edu/ml/datasets/heart+Disease
  Also : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
  Rows : 303 clinical records collected 1988 at Cleveland Clinic.
  Features (13 input + 1 target):
    age, sex, cp (chest pain type), trestbps (resting BP),
    chol (serum cholesterol), fbs (fasting blood sugar > 120),
    restecg (resting ECG), thalach (max HR), exang (exercise angina),
    oldpeak (ST depression), slope, ca (coloured vessels 0–4), thal
  Target : 0 = no disease, 1–4 = disease severity (we binarise to 0/1)

Extended datasets (optional):
  • PhysioNet PTB-XL  – raw 12-lead ECG waveforms (21 837 records)
    https://physionet.org/content/ptb-xl/1.0.3/
  • Kaggle Arrhythmia  – MIT-BIH Arrhythmia Database
    https://www.kaggle.com/datasets/mondejar/mitbih-database
────────────────────────────────────────────────────────────────────

This script:
  1. Downloads the Cleveland dataset automatically via ucimlrepo.
  2. Cleans / encodes the data.
  3. Trains a Random-Forest + calibrated probability model.
  4. Evaluates on a hold-out test set.
  5. Saves the model to app/ml_models/cardiocode_rf_model.pkl
  6. Saves a SHAP explainer to app/ml_models/shap_explainer.pkl

Run once:
  python app/ml_models/training/train_model.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ─── paths ────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = BASE                                # save models here
MODEL_PATH    = os.path.join(OUT, "cardiocode_rf_model.pkl")
EXPLAINER_PATH = os.path.join(OUT, "shap_explainer.pkl")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

def load_data() -> pd.DataFrame:
    """Download or load the UCI Heart Disease (Cleveland) dataset."""
    try:
        # Try fetching via ucimlrepo (pip install ucimlrepo)
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()
    except Exception:
        # Fallback: download raw CSV from UCI
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        cols = FEATURE_COLS + ["target"]
        df = pd.read_csv(url, names=cols, na_values="?")
        X = df[FEATURE_COLS]
        y = df["target"]

    # Binarise: 0 = no disease, 1 = disease
    y = (y > 0).astype(int)
    df = X.copy()
    df["target"] = y
    return df


def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    base = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   CalibratedClassifierCV(base, cv=5, method="sigmoid")),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== CardioCode Model Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["No Disease","Disease"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def build_shap_explainer(model, X_train):
    """Build a TreeExplainer on the calibrated base estimator."""
    try:
        calibrated_clf = model.named_steps["model"]
        inner_model    = calibrated_clf.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(inner_model)
    except Exception:
        # Fallback generic explainer
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_train, 50)
        )
    return explainer


def main():
    print("[Training] Loading UCI Heart Disease dataset...")
    df = load_data()
    print(f"   Rows: {len(df)}  |  Disease prevalence: {df['target'].mean():.1%}")

    X_train, X_test, y_train, y_test = preprocess(df)
    print("[Training] Fitting Gradient Boosting + Calibration pipeline...")
    model = train(X_train, y_train)

    evaluate(model, X_test, y_test)

    print("[Training] Saving model...")
    joblib.dump(model, MODEL_PATH)
    print(f"   -> {MODEL_PATH}")

    print("[Training] Building SHAP explainer...")
    imputed_X_train = model.named_steps["imputer"].transform(X_train)
    imputed_df      = pd.DataFrame(imputed_X_train, columns=FEATURE_COLS)
    explainer       = build_shap_explainer(model, imputed_df)
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"   -> {EXPLAINER_PATH}")

    print("\n[Training] Complete - model saved successfully.")


if __name__ == "__main__":
    main()
