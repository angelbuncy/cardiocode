# ============================================================
#  CardioCode – Firebase Firestore Database Layer
#  Replaces SQLAlchemy/Postgres.  All collections are created
#  on first write – no migrations needed.
# ============================================================
import os
import firebase_admin
from firebase_admin import credentials, firestore

_app = None

def get_firestore_client():
    """
    Returns an authenticated Firestore client.
    On first call it initialises the Firebase Admin SDK using the
    service-account JSON pointed to by FIREBASE_CREDENTIALS_PATH.

    How to obtain the JSON:
      Firebase Console ➜ Project Settings ➜ Service Accounts
      ➜ Generate new private key  ➜ save as
        D:\\PROJECTS\\Cardiocode\\backend\\firebase-credentials.json
      then set:
        FIREBASE_CREDENTIALS_PATH=D:\\PROJECTS\\Cardiocode\\backend\\firebase-credentials.json
    """
    global _app
    if _app is None:
        cred_path = os.getenv(
            "FIREBASE_CREDENTIALS_PATH",
            "firebase-credentials.json"
        )
        cred = credentials.Certificate(cred_path)
        _app = firebase_admin.initialize_app(cred)
    return firestore.client()


# ─── Collection name constants ────────────────────────────────
USERS_COL          = "users"
PATIENTS_COL       = "patient_records"
TRIAGE_COL         = "triage_results"
APPOINTMENTS_COL   = "appointments"
HOSPITALS_COL      = "hospitals"
