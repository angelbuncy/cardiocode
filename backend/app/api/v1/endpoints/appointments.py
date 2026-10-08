"""
CardioCode – Appointment Scheduling Router
POST /api/v1/appointments/schedule
GET  /api/v1/appointments/{patient_id}
PUT  /api/v1/appointments/{appt_id}/status
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services import firebase_service as fb

router = APIRouter()


class ScheduleRequest(BaseModel):
    patient_record_id: str
    doctor_uid:        str
    hospital_id:       str
    triage_result_id:  str
    preferred_time:    Optional[str] = None   # ISO datetime string
    channel:           str = "In-Person"      # In-Person | Teleconsult
    notes:             str = ""


class StatusUpdate(BaseModel):
    status: str   # Confirmed | Completed | Cancelled


@router.post("/schedule")
async def schedule_appointment(data: ScheduleRequest):
    """
    Create a new appointment.
    If triage shows High/Critical risk and no preferred_time given,
    the slot is automatically set to next available (within 24 hrs).
    """
    # Check triage result
    triage = fb.get_triage_result(data.triage_result_id)
    risk_level = triage.get("risk_level", "Low") if triage else "Low"

    priority = risk_level in ("High", "Critical")
    # Auto-assign time if not provided
    appt_time = data.preferred_time or (
        datetime.utcnow() + timedelta(hours=2 if priority else 48)
    ).isoformat()

    appt_data = {
        "patient_record_id": data.patient_record_id,
        "doctor_uid":        data.doctor_uid,
        "hospital_id":       data.hospital_id,
        "triage_result_id":  data.triage_result_id,
        "appointment_time":  appt_time,
        "priority":          priority,
        "status":            "Scheduled",
        "channel":           data.channel,
        "notes":             data.notes,
    }

    appt_id = fb.create_appointment(appt_data)

    return {
        "appointment_id":   appt_id,
        "status":           "Scheduled",
        "priority":         priority,
        "appointment_time": appt_time,
        "channel":          data.channel,
        "message": (
            "⚠️ Priority slot allocated – patient flagged High/Critical Risk."
            if priority else
            "Appointment scheduled successfully."
        ),
    }


@router.get("/{patient_id}")
async def list_appointments(patient_id: str):
    """List all appointments for a patient."""
    appts = fb.list_appointments_by_patient(patient_id)
    return {"patient_id": patient_id, "appointments": appts}


@router.put("/{appt_id}/status")
async def update_status(appt_id: str, body: StatusUpdate):
    """Update appointment status (Confirmed / Completed / Cancelled)."""
    fb.update_appointment_status(appt_id, body.status)
    return {"appointment_id": appt_id, "new_status": body.status}
