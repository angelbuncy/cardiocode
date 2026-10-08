"""
CardioCode – Firebase Firestore Service Layer
All CRUD helpers for every collection.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.db.database import (
    get_firestore_client,
    USERS_COL, PATIENTS_COL, TRIAGE_COL, APPOINTMENTS_COL, HOSPITALS_COL
)
from app.models.schemas import (
    UserDoc, PatientRecordDoc, TriageResultDoc, AppointmentDoc, HospitalDoc, to_dict
)


def _new_id() -> str:
    return str(uuid.uuid4())


# ─── HOSPITAL ────────────────────────────────────────────────────────────────
def create_hospital(data: dict) -> str:
    db = get_firestore_client()
    doc_id = _new_id()
    doc = HospitalDoc(id=doc_id, **data)
    db.collection(HOSPITALS_COL).document(doc_id).set(to_dict(doc))
    return doc_id

def get_hospital(hospital_id: str) -> Optional[dict]:
    db = get_firestore_client()
    doc = db.collection(HOSPITALS_COL).document(hospital_id).get()
    return doc.to_dict() if doc.exists else None

def list_hospitals() -> List[dict]:
    db = get_firestore_client()
    return [d.to_dict() for d in db.collection(HOSPITALS_COL).stream()]


# ─── USERS ────────────────────────────────────────────────────────────────────
def create_user(data: dict) -> str:
    db = get_firestore_client()
    doc = UserDoc(**data)
    db.collection(USERS_COL).document(doc.uid).set(to_dict(doc))
    return doc.uid

def get_user(uid: str) -> Optional[dict]:
    db = get_firestore_client()
    doc = db.collection(USERS_COL).document(uid).get()
    return doc.to_dict() if doc.exists else None

def get_user_by_email(email: str) -> Optional[dict]:
    db = get_firestore_client()
    docs = db.collection(USERS_COL).where("email", "==", email).limit(1).stream()
    for d in docs:
        return d.to_dict()
    return None


# ─── PATIENT RECORDS ──────────────────────────────────────────────────────────
def create_patient(data: dict) -> str:
    db = get_firestore_client()
    doc_id = _new_id()
    doc = PatientRecordDoc(id=doc_id, **data)
    db.collection(PATIENTS_COL).document(doc_id).set(to_dict(doc))
    return doc_id

def get_patient(patient_id: str) -> Optional[dict]:
    db = get_firestore_client()
    doc = db.collection(PATIENTS_COL).document(patient_id).get()
    return doc.to_dict() if doc.exists else None

def update_patient(patient_id: str, updates: dict) -> bool:
    db = get_firestore_client()
    updates["updated_at"] = datetime.utcnow().isoformat()
    db.collection(PATIENTS_COL).document(patient_id).update(updates)
    return True

def list_patients_by_hospital(hospital_id: str) -> List[dict]:
    db = get_firestore_client()
    return [d.to_dict() for d in
            db.collection(PATIENTS_COL).where("hospital_id", "==", hospital_id).stream()]


# ─── TRIAGE RESULTS ───────────────────────────────────────────────────────────
def create_triage_result(data: dict) -> str:
    db = get_firestore_client()
    doc_id = _new_id()
    doc = TriageResultDoc(id=doc_id, **data)
    db.collection(TRIAGE_COL).document(doc_id).set(to_dict(doc))
    return doc_id

def get_triage_result(triage_id: str) -> Optional[dict]:
    db = get_firestore_client()
    doc = db.collection(TRIAGE_COL).document(triage_id).get()
    return doc.to_dict() if doc.exists else None

def list_triage_by_patient(patient_id: str) -> List[dict]:
    db = get_firestore_client()
    return [d.to_dict() for d in
            db.collection(TRIAGE_COL)
              .where("patient_record_id", "==", patient_id)
              .order_by("created_at", direction="DESCENDING")
              .stream()]


# ─── APPOINTMENTS ─────────────────────────────────────────────────────────────
def create_appointment(data: dict) -> str:
    db = get_firestore_client()
    doc_id = _new_id()
    doc = AppointmentDoc(id=doc_id, **data)
    db.collection(APPOINTMENTS_COL).document(doc_id).set(to_dict(doc))
    return doc_id

def update_appointment_status(appt_id: str, status: str) -> bool:
    db = get_firestore_client()
    db.collection(APPOINTMENTS_COL).document(appt_id).update({"status": status})
    return True

def list_appointments_by_patient(patient_id: str) -> List[dict]:
    db = get_firestore_client()
    return [d.to_dict() for d in
            db.collection(APPOINTMENTS_COL)
              .where("patient_record_id", "==", patient_id)
              .stream()]
