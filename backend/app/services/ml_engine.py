"""
CardioCode – ML Inference Engine
Loads the trained model + SHAP explainer at startup (singleton).
Provides predict() which returns risk_score, risk_level, and per-feature
SHAP importance values so doctors can audit the prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Tuple

from app.core.config import settings

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar >120 mg/dL",
    "restecg":  "Resting ECG Results (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina (1=Yes)",
    "oldpeak":  "ST Depression (exercise vs rest)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "Major Vessels Coloured by Fluoroscopy (0-4)",
    "thal":     "Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible)",
}

# ─── Singleton ────────────────────────────────────────────────────
_model = None
_explainer = None
_model_loaded = False


def _load():
    global _model, _explainer, _model_loaded
    if _model_loaded:
        return

    model_path = settings.MODEL_PATH
    explainer_path = model_path.replace("cardiocode_rf_model.pkl", "shap_explainer.pkl")

    if os.path.exists(model_path):
        _model = joblib.load(model_path)
        if os.path.exists(explainer_path):
            _explainer = joblib.load(explainer_path)
        _model_loaded = True
        print(f"[CardioCode] ML Model loaded from {model_path}")
    else:
        print("[CardioCode] Model not found - using fallback heuristic scoring.")
        _model_loaded = True  # Don't try again


def _heuristic_score(features: dict) -> float:
    """Simple evidence-based fallback when model hasn't been trained yet."""
    score = 0.0
    if features.get("age", 0) > 55:      score += 0.15
    if features.get("sex", 0) == 1:       score += 0.10
    if features.get("cp", 0) in [1, 2]:   score += 0.20
    if features.get("trestbps", 0) > 140: score += 0.10
    if features.get("chol", 0) > 240:     score += 0.10
    if features.get("exang", 0) == 1:     score += 0.15
    if features.get("thalach", 0) < 130:  score += 0.10
    if features.get("oldpeak", 0) > 2.0:  score += 0.10
    return min(score, 1.0)


def _risk_level(score: float) -> str:
    if score >= 0.75:  return "Critical"
    if score >= 0.50:  return "High"
    if score >= 0.25:  return "Moderate"
    return "Low"


def predict(features: dict) -> Tuple[float, str, Dict[str, float]]:
    """
    Returns (risk_score, risk_level, feature_importance_dict)
    feature_importance_dict maps feature name → SHAP value (% contribution)
    """
    _load()

    row = pd.DataFrame([[features.get(c, 0) for c in FEATURE_COLS]],
                       columns=FEATURE_COLS)

    if _model is None:
        score = _heuristic_score(features)
        importance = {
            c: round(abs(features.get(c, 0)) / (sum(abs(features.get(c, 0)) for c in FEATURE_COLS) + 1e-9), 4)
            for c in FEATURE_COLS
        }
    else:
        score = float(_model.predict_proba(row)[0, 1])

        if _explainer is not None:
            try:
                imputed = _model.named_steps["imputer"].transform(row)
                shap_vals = _explainer.shap_values(imputed)
                # For GBT the output is a 1-D array
                if isinstance(shap_vals, list):
                    vals = np.abs(shap_vals[1][0])
                else:
                    vals = np.abs(shap_vals[0])
                total = vals.sum() or 1.0
                importance = {
                    FEATURE_COLS[i]: round(float(vals[i] / total), 4)
                    for i in range(len(FEATURE_COLS))
                }
            except Exception:
                importance = {c: 0.0 for c in FEATURE_COLS}
        else:
            importance = {c: 0.0 for c in FEATURE_COLS}

    level = _risk_level(score)
    return round(score, 4), level, importance
