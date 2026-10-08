"""
CardioCode – Patient Records Router
POST /api/v1/patients/
GET  /api/v1/patients/{patient_id}
PUT  /api/v1/patients/{patient_id}
GET  /api/v1/patients/hospital/{hospital_id}
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.services import firebase_service as fb

router = APIRouter()


class CreatePatientRequest(BaseModel):
    user_uid:       str
    hospital_id:    str
    full_name:      str
    dob:            str          # "YYYY-MM-DD"
    sex:            str          # "M" | "F" | "Other"
    blood_group:    str
    contact_phone:  str
    contact_email:  str
    emergency_contact: str
    allergies:      List[str] = []
    chronic_conditions: List[str] = []
    current_medications: List[str] = []
    previous_surgeries: List[str] = []
    family_history: Dict[str, Any] = {}
    insurance_provider:  Optional[str] = None
    insurance_policy_no: Optional[str] = None


class UpdatePatientRequest(BaseModel):
    chronic_conditions:   Optional[List[str]] = None
    current_medications:  Optional[List[str]] = None
    allergies:            Optional[List[str]] = None
    emergency_contact:    Optional[str] = None
    contact_phone:        Optional[str] = None
    insurance_provider:   Optional[str] = None
    insurance_policy_no:  Optional[str] = None


@router.post("/")
async def create_patient(data: CreatePatientRequest):
    patient_id = fb.create_patient(data.dict())
    return {"patient_id": patient_id, "message": "Patient record created."}


@router.get("/{patient_id}")
async def get_patient(patient_id: str):
    patient = fb.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


@router.put("/{patient_id}")
async def update_patient(patient_id: str, data: UpdatePatientRequest):
    updates = {k: v for k, v in data.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided.")
    fb.update_patient(patient_id, updates)
    return {"patient_id": patient_id, "updated_fields": list(updates.keys())}


@router.get("/hospital/{hospital_id}")
async def list_by_hospital(hospital_id: str):
    patients = fb.list_patients_by_hospital(hospital_id)
    return {"hospital_id": hospital_id, "patients": patients, "count": len(patients)}
