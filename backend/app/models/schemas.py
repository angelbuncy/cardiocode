"""
CardioCode – Firestore Document Schemas
(Python dataclasses acting as typed schema contracts)

Firestore is schema-less; these classes document the shape of
every document stored in each collection.

Collections
───────────
users/               – accounts (Admin | Doctor | Patient)
hospitals/           – hospital / diagnostic centre metadata
patient_records/     – demographics + medical history
triage_results/      – ML prediction outputs per visit
appointments/        – scheduled consultations
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any


# ──────────────────────────────────────────────────────────────────
# users/  {uid}
# ──────────────────────────────────────────────────────────────────
@dataclass
class UserDoc:
    uid: str                          # Firebase Auth UID
    email: str
    full_name: str
    role: str                         # "Admin" | "Doctor" | "Patient"
    hospital_id: Optional[str] = None # FK → hospitals/{id}
    phone: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_active: bool = True


# ──────────────────────────────────────────────────────────────────
# hospitals/  {id}
# ──────────────────────────────────────────────────────────────────
@dataclass
class HospitalDoc:
    id: str
    name: str
    address: str
    city: str
    state: str
    country: str
    phone: str
    email: str
    accreditation: str                # e.g. "NABH", "JCI"
    specializations: List[str] = field(default_factory=list)
    doctor_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────────────
# patient_records/  {id}
# ──────────────────────────────────────────────────────────────────
@dataclass
class PatientRecordDoc:
    id: str
    user_uid: str                     # FK → users/{uid}
    hospital_id: str                  # FK → hospitals/{id}

    # demographics
    full_name: str
    dob: str                          # ISO date "YYYY-MM-DD"
    sex: str                          # "M" | "F" | "Other"
    blood_group: str
    contact_phone: str
    contact_email: str
    emergency_contact: str

    # medical history (free-form JSON)
    allergies: List[str] = field(default_factory=list)
    chronic_conditions: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    previous_surgeries: List[str] = field(default_factory=list)
    family_history: Dict[str, Any] = field(default_factory=dict)

    # insurance
    insurance_provider: Optional[str] = None
    insurance_policy_no: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────────────
# triage_results/  {id}
# ──────────────────────────────────────────────────────────────────
@dataclass
class TriageResultDoc:
    id: str
    patient_record_id: str            # FK → patient_records/{id}
    doctor_uid: str                   # FK → users/{uid}
    hospital_id: str                  # FK → hospitals/{id}

    # raw clinical inputs
    age: int
    sex: int                          # 0=F, 1=M
    cp: int                           # chest-pain type 0-3
    trestbps: int                     # resting blood pressure
    chol: int                         # serum cholesterol mg/dL
    fbs: int                          # fasting blood sugar > 120 mg/dL
    restecg: int                      # resting ECG 0-2
    thalach: int                      # max heart rate achieved
    exang: int                        # exercise-induced angina
    oldpeak: float                    # ST depression
    slope: int                        # slope of peak ST segment
    ca: int                           # major vessels coloured 0-4
    thal: int                         # thalassemia 0-3

    # ML outputs
    risk_score: float                 # 0.0 – 1.0
    risk_level: str                   # "Low" | "Moderate" | "High" | "Critical"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0.0"
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────────────
# appointments/  {id}
# ──────────────────────────────────────────────────────────────────
@dataclass
class AppointmentDoc:
    id: str
    patient_record_id: str            # FK → patient_records/{id}
    doctor_uid: str                   # FK → users/{uid}
    hospital_id: str                  # FK → hospitals/{id}
    triage_result_id: str             # FK → triage_results/{id}

    appointment_time: str             # ISO datetime
    status: str = "Scheduled"         # Scheduled | Confirmed | Completed | Cancelled
    priority: bool = False            # True if High/Critical risk
    channel: str = "In-Person"        # In-Person | Teleconsult
    notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def to_dict(doc) -> Dict:
    return asdict(doc)
