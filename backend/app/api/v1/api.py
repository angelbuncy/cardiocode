from fastapi import APIRouter
from app.api.v1.endpoints import triage, appointments, alerts, patients, hospitals

api_router = APIRouter()
api_router.include_router(triage.router,      prefix="/triage",       tags=["Triage / ML"])
api_router.include_router(appointments.router, prefix="/appointments", tags=["Appointments"])
api_router.include_router(alerts.router,       prefix="/alerts",       tags=["Real-Time Alerts"])
api_router.include_router(patients.router,     prefix="/patients",     tags=["Patient Records"])
api_router.include_router(hospitals.router,    prefix="/hospitals",    tags=["Hospital Metadata"])
