from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from datetime import datetime

class PatientProfile(BaseModel):
    patient_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    last_login: Optional[str] = None
    medical_data: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "0006ad41-c2d3-4994-8aab-7a3a107d50aa",
                "email": "john.smith@example.com",
                "first_name": "John",
                "last_name": "Smith",
                "date_of_birth": "1990-01-15",
                "gender": "male",
                "last_login": "2025-05-15T12:34:56.789Z",
                "medical_data": {
                    "first_name": "John",
                    "last_name": "Smith",
                    "date_of_birth": "1990-01-15",
                    "gender": "male",
                    "city": "Boston",
                    "state": "MA"
                }
            }
        }

class Symptom(BaseModel):
    symptom_id: Optional[int] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    factor: Optional[str] = None
    severity: Optional[float] = None
    symptom_date: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symptom_id": 123,
                "category": "Cognitive",
                "subcategory": "Memory",
                "factor": "Stress",
                "severity": 7.5,
                "symptom_date": "2025-05-10"
            }
        }

class SymptomList(BaseModel):
    symptoms: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "symptoms": [
                    {
                        "symptom_id": 123,
                        "category": "Cognitive",
                        "subcategory": "Memory",
                        "factor": "Stress",
                        "severity": 7.5,
                        "symptom_date": "2025-05-10"
                    },
                    {
                        "symptom_id": 124,
                        "category": "Physical",
                        "subcategory": "Headache",
                        "factor": "Noise",
                        "severity": 6.0,
                        "symptom_date": "2025-05-12"
                    }
                ]
            }
        }