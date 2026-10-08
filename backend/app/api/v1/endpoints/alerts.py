"""
CardioCode – Real-Time Alerts via Server-Sent Events (SSE)
GET /api/v1/alerts/stream

The doctor dashboard connects to this endpoint and receives live alerts
whenever a new triage result with High or Critical risk is written to Firestore.

Firestore on_snapshot listener pushes events into an async queue.
The SSE generator pulls from that queue and yields to the client.
"""
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.db.database import get_firestore_client, TRIAGE_COL

router = APIRouter()

# Global queue for critical alerts
_alert_queue: asyncio.Queue = asyncio.Queue()
_listener_started = False


def _start_firestore_listener():
    """Register a Firestore on_snapshot callback for new critical triage docs."""
    global _listener_started
    if _listener_started:
        return
    _listener_started = True

    db = get_firestore_client()

    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if change.type.name == "ADDED":
                doc = change.document.to_dict()
                if doc.get("risk_level") in ("High", "Critical"):
                    alert = {
                        "triage_id":    doc.get("id", ""),
                        "patient_id":   doc.get("patient_record_id", ""),
                        "risk_level":   doc.get("risk_level", ""),
                        "risk_score":   doc.get("risk_score", 0.0),
                        "timestamp":    doc.get("created_at", datetime.utcnow().isoformat()),
                        "message":      f"🚨 {doc.get('risk_level')} cardiac risk detected!",
                    }
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(_alert_queue.put_nowait, alert)
                    except Exception:
                        pass   # Event loop may not be running yet

    db.collection(TRIAGE_COL).on_snapshot(on_snapshot)


@router.get("/stream")
async def stream_alerts(request: Request, hospital_id: str = ""):
    """
    SSE endpoint – connects to the doctor dashboard and streams real-time
    High/Critical triage alerts as they arrive in Firestore.

    Usage (frontend):
        const es = new EventSource('/api/v1/alerts/stream?hospital_id=XYZ');
        es.addEventListener('critical_alert', e => console.log(JSON.parse(e.data)));
    """
    _start_firestore_listener()

    async def generator():
        # Heartbeat every 15 s to keep connection alive
        yield {"event": "connected", "data": json.dumps({"status": "Listening for alerts..."})}
        while True:
            if await request.is_disconnected():
                break
            try:
                alert = await asyncio.wait_for(_alert_queue.get(), timeout=15.0)
                yield {
                    "event": "critical_alert",
                    "id":    alert["triage_id"],
                    "data":  json.dumps(alert),
                }
            except asyncio.TimeoutError:
                # heartbeat ping
                yield {"event": "ping", "data": json.dumps({"ts": datetime.utcnow().isoformat()})}

    return EventSourceResponse(generator())
