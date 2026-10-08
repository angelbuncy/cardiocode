"""
CardioCode – Triage / ML Inference Router
POST /api/v1/triage/predict
GET  /api/v1/triage/history/{patient_id}
"""
import uuid
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import ml_engine, firebase_service as fb

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────
class TriageRequest(BaseModel):
    patient_record_id: str
    doctor_uid:        str
    hospital_id:       str

    # Clinical features (13 standard UCI Heart Disease features)
    age:      int   = Field(..., ge=1, le=120, description="Patient age in years")
    sex:      int   = Field(..., ge=0, le=1,   description="1=Male, 0=Female")
    cp:       int   = Field(..., ge=0, le=3,   description="Chest pain type 0-3")
    trestbps: int   = Field(..., ge=80, le=220, description="Resting BP (mmHg)")
    chol:     int   = Field(..., ge=100, le=600, description="Serum Cholesterol mg/dL")
    fbs:      int   = Field(..., ge=0, le=1,   description="Fasting blood sugar > 120 mg/dL")
    restecg:  int   = Field(..., ge=0, le=2,   description="Resting ECG result 0-2")
    thalach:  int   = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0, le=1,   description="Exercise-induced angina 1=Yes")
    oldpeak:  float = Field(..., ge=0.0, le=10.0, description="ST depression")
    slope:    int   = Field(..., ge=0, le=2,   description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0, le=4,   description="Major vessels coloured (0-4)")
    thal:     int   = Field(..., ge=0, le=3,   description="Thalassemia 0-3")
    notes:    str   = ""


class TriageResponse(BaseModel):
    triage_id:         str
    risk_score:        float
    risk_level:        str
    feature_importance: Dict[str, float]
    recommendation:    str
    created_at:        str


# ─── Helpers ──────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Low":      "Routine follow-up recommended. Maintain healthy lifestyle.",
    "Moderate": "Schedule cardiology review within 2 weeks. Monitor BP and cholesterol.",
    "High":     "Priority appointment required within 48 hours. Cardiologist review needed.",
    "Critical": "IMMEDIATE medical attention required. Emergency triage activated.",
}


# ─── Routes ───────────────────────────────────────────────────────
@router.post("/predict", response_model=TriageResponse)
async def predict_triage(data: TriageRequest):
    """
    Run ML cardiac risk prediction.
    Saves result to Firestore and returns risk score + SHAP feature importance.
    """
    # Verify patient exists
    patient = fb.get_patient(data.patient_record_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found")

    # Run inference
    features = data.dict(exclude={"patient_record_id", "doctor_uid", "hospital_id", "notes"})
    risk_score, risk_level, feature_importance = ml_engine.predict(features)

    # Store in Firestore
    triage_doc = {
        "patient_record_id": data.patient_record_id,
        "doctor_uid":        data.doctor_uid,
        "hospital_id":       data.hospital_id,
        "risk_score":        risk_score,
        "risk_level":        risk_level,
        "feature_importance": feature_importance,
        "notes":             data.notes,
        **features
    }
    triage_id = fb.create_triage_result(triage_doc)

    # Auto-schedule if High/Critical
    if risk_level in ("High", "Critical"):
        appt = {
            "patient_record_id": data.patient_record_id,
            "doctor_uid":        data.doctor_uid,
            "hospital_id":       data.hospital_id,
            "triage_result_id":  triage_id,
            "appointment_time":  datetime.utcnow().isoformat(),
            "priority":          True,
            "status":            "Auto-Scheduled (Priority)",
        }
        fb.create_appointment(appt)

    return TriageResponse(
        triage_id=triage_id,
        risk_score=risk_score,
        risk_level=risk_level,
        feature_importance=feature_importance,
        recommendation=RECOMMENDATIONS[risk_level],
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/history/{patient_id}")
async def get_triage_history(patient_id: str):
    """Retrieve all triage results for a patient (chronological)."""
    results = fb.list_triage_by_patient(patient_id)
    if not results:
        return {"patient_id": patient_id, "history": []}
    return {"patient_id": patient_id, "history": results}
