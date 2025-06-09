import logging
from typing import Optional, Dict, Any, List
import json
import datetime
from core.database import get_db_connection
import psycopg2.extras
from utils.helpers import generate_verification_code, format_date_iso
from core.email import send_verification_email, send_welcome_email
from core.config import EMAIL_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

def create_or_update_verified_patient(patient_id: str, email: str, verification_data: Dict[str, Any]) -> bool:
    """
    Create or update a verified patient record
    
    Args:
        patient_id: Patient ID
        email: Email address
        verification_data: Additional verification data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate verification code
        verification_code = generate_verification_code()
        expiry_hours = EMAIL_CONFIG.get('verification_expiry_hours', 24)
        verification_expiry = datetime.datetime.now() + datetime.timedelta(hours=expiry_hours)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get patient information
        patient_info = verification_data.get('patient_info', {})
        first_name = patient_info.get('first_name', '')
        last_name = patient_info.get('last_name', '')
        dob = patient_info.get('date_of_birth', None)
        gender = patient_info.get('gender', None)
        google_id = verification_data.get('google_id', None)
        password_hash = verification_data.get('password_hash', None)
        
        # Check if patient already exists
        cursor.execute("SELECT id FROM verified_patients WHERE patient_id = %s", (patient_id,))
        patient_exists = cursor.fetchone() is not None
        
        if patient_exists:
            # Update existing patient
            update_fields = []
            params = []
            
            # Only update non-None fields
            if email:
                update_fields.append("email = %s")
                params.append(email)
            if first_name:
                update_fields.append("first_name = %s")
                params.append(first_name)
            if last_name:
                update_fields.append("last_name = %s")
                params.append(last_name)
            if dob:
                update_fields.append("date_of_birth = %s")
                params.append(dob)
            if gender is not None:
                update_fields.append("gender = %s")
                params.append(gender)
            if google_id:
                update_fields.append("google_id = %s")
                params.append(google_id)
            if password_hash:
                update_fields.append("password_hash = %s")
                params.append(password_hash)
                
            # Always update verification fields
            update_fields.extend([
                "verification_code = %s",
                "verification_expiry = %s",
                "is_verified = FALSE",
                "updated_at = NOW()"
            ])
            params.extend([verification_code, verification_expiry])
            
            # Add patient_id to params
            params.append(patient_id)
            
            # Execute update
            cursor.execute(f"""
                UPDATE verified_patients
                SET {', '.join(update_fields)}
                WHERE patient_id = %s
            """, params)
            
        else:
            # Create new patient
            cursor.execute("""
                INSERT INTO verified_patients
                (patient_id, email, first_name, last_name, date_of_birth, gender, google_id, 
                 password_hash, is_verified, verification_code, verification_expiry)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s, %s)
            """, (patient_id, email, first_name, last_name, dob, gender, google_id, 
                  password_hash, verification_code, verification_expiry))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Send verification email
        if email and first_name:
            result = send_verification_email(email, first_name, verification_code)
            if not result:
                logger.error(f"Failed to send verification email to {email}")
                return False
        
        return True
        
    except Exception as e:
        logger.exception(f"Error creating/updating verified patient: {str(e)}")
        return False

def verify_code(patient_id: str, verification_code: str) -> bool:
    """
    Verify a patient's verification code
    
    Args:
        patient_id: Patient ID
        verification_code: Verification code
        
    Returns:
        bool: True if verification successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Verify code and check expiry
        cursor.execute("""
            SELECT verification_code, verification_expiry, email, first_name
            FROM verified_patients
            WHERE patient_id = %s AND is_verified = FALSE
        """, (patient_id,))
        
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            logger.warning(f"No pending verification found for patient ID: {patient_id}")
            return False
        
        stored_code, expiry, email, first_name = result['verification_code'], result['verification_expiry'], result['email'], result['first_name']
        
        # Check if code matches and is not expired
        if stored_code == verification_code and expiry > datetime.datetime.now():
            # Mark as verified
            cursor.execute("""
                UPDATE verified_patients
                SET is_verified = TRUE,
                    verification_code = NULL,
                    verification_expiry = NULL,
                    last_login = NOW(),
                    updated_at = NOW()
                WHERE patient_id = %s
            """, (patient_id,))
            
            conn.commit()
            
            # Load patient data into context
            from services.patient_context import load_patient_context
            load_patient_context(patient_id)
            
            cursor.close()
            conn.close()
            
            # Send welcome email
            send_welcome_email(email, first_name)
            
            return True
        else:
            cursor.close()
            conn.close()
            logger.warning(f"Invalid or expired verification code for patient ID: {patient_id}")
            return False
        
    except Exception as e:
        logger.exception(f"Error verifying code: {str(e)}")
        return False

def get_patient_by_email(email: str) -> Optional[Dict]:
    """
    Get a patient by email
    
    Args:
        email: Email address
        
    Returns:
        Optional[Dict]: Patient data if found, None otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute("""
            SELECT patient_id, password_hash, is_verified, first_name
            FROM verified_patients
            WHERE email = %s
        """, (email,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return dict(result)
        return None
        
    except Exception as e:
        logger.exception(f"Error getting patient by email: {str(e)}")
        return None

def update_last_login(patient_id: str) -> bool:
    """
    Update a patient's last login timestamp
    
    Args:
        patient_id: Patient ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE verified_patients
            SET last_login = NOW()
            WHERE patient_id = %s
        """, (patient_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.exception(f"Error updating last login: {str(e)}")
        return False