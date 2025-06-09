import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
    'dbname': os.getenv("DB_NAME")
}

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': os.getenv("SMTP_SERVER"),
    'smtp_port': int(os.getenv("SMTP_PORT", 587)),
    'smtp_user': os.getenv("SMTP_USER"),
    'smtp_password': os.getenv("SMTP_PASSWORD"),
    'from_email': os.getenv("FROM_EMAIL"),
    'verification_expiry_hours': int(os.getenv("VERIFICATION_EXPIRY_HOURS", 24))
}

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET") or os.urandom(24).hex()
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", 24 * 60 * 60))  # 24 hours in seconds
JWT_ALGORITHM = "HS256"

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")