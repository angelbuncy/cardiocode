"""
CardioCode – Environment / Settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "CardioCode API"
    VERSION: str = "1.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PROD_super_secret")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 hours

    FIREBASE_CREDENTIALS_PATH: str = os.getenv(
        "FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json"
    )

    # Redis / Celery (optional)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Twilio (SMS notifications)
    TWILIO_SID: str = os.getenv("TWILIO_SID", "")
    TWILIO_TOKEN: str = os.getenv("TWILIO_TOKEN", "")
    TWILIO_FROM: str = os.getenv("TWILIO_FROM", "")

    # SendGrid (Email)
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "no-reply@cardiocode.app")

    # ML Model
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", "app/ml_models/cardiocode_rf_model.pkl"
    )


settings = Settings()
