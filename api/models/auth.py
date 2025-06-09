from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any, Dict, List

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    patient_id: Optional[str] = None

class GoogleAuthRequest(BaseModel):
    code: str

class VerificationRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None
    patient_info: Optional[str] = None
    email: EmailStr

class CompleteRegistrationRequest(BaseModel):
    first_name: str
    last_name: str
    dob: str
    gender: Optional[str] = None
    google_id: Optional[str] = None
    google_email: Optional[EmailStr] = None

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    dob: str
    gender: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Smith",
                "email": "john.smith@example.com",
                "password": "securepassword123",
                "dob": "1990-01-15",
                "gender": "male"
            }
        }

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "john.smith@example.com",
                "password": "securepassword123"
            }
        }

class VerifyCodeRequest(BaseModel):
    patient_id: str
    verification_code: str
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "0006ad41-c2d3-4994-8aab-7a3a107d50aa",
                "verification_code": "123456"
            }
        }

class RegisterResponse(BaseModel):
    message: str
    patient_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Registration successful. Please check your email for a verification code.",
                "patient_id": "0006ad41-c2d3-4994-8aab-7a3a107d50aa"
            }
        }

class VerifyPatientResponse(BaseModel):
    message: str
    patient_id: str
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Verification email sent. Please check your email for a verification code.",
                "patient_id": "0006ad41-c2d3-4994-8aab-7a3a107d50aa"
            }
        }