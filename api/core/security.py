import jwt
from datetime import datetime, timezone, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import hashlib
import os

from core.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwd_hash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwd_hash = hashlib.sha256(pwd_hash).hexdigest()
    return (salt + pwd_hash).decode('ascii')

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_pwd_hash = stored_password[64:]
    pwd_hash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwd_hash = hashlib.sha256(pwd_hash).hexdigest()
    return pwd_hash == stored_pwd_hash

def create_jwt_token(patient_id: str) -> str:
    """Create a JWT token for a patient"""
    payload = {
        "sub": patient_id,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRATION),
        "iat": datetime.now(timezone.utc)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt_token(token: str) -> str:
    """Decode a JWT token and return the patient_id if valid"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return patient_id"""
    token = credentials.credentials
    patient_id = decode_jwt_token(token)
    if not patient_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return patient_id