"""
CardioCode – Hospital Metadata Router
POST /api/v1/hospitals/
GET  /api/v1/hospitals/
GET  /api/v1/hospitals/{hospital_id}
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import firebase_service as fb

router = APIRouter()


class CreateHospitalRequest(BaseModel):
    name:            str
    address:         str
    city:            str
    state:           str
    country:         str
    phone:           str
    email:           str
    accreditation:   str
    specializations: List[str] = []


@router.post("/")
async def create_hospital(data: CreateHospitalRequest):
    hosp_id = fb.create_hospital(data.dict())
    return {"hospital_id": hosp_id, "message": "Hospital registered."}


@router.get("/")
async def list_hospitals():
    hospitals = fb.list_hospitals()
    return {"hospitals": hospitals, "count": len(hospitals)}


@router.get("/{hospital_id}")
async def get_hospital(hospital_id: str):
    h = fb.get_hospital(hospital_id)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    return h
