import logging
from typing import Dict, Any, Optional, List
import json
import datetime
from core.database import get_db_connection
import psycopg2.extras

# Configure logging
logger = logging.getLogger(__name__)

def load_patient_context(patient_id: str) -> bool:
    """
    Load patient context data from patients and symptom_logs tables
    
    Args:
        patient_id: Patient ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get patient summary data
        cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
        summary_result = cursor.fetchone()
        
        if not summary_result:
            logger.warning(f"No patient found with ID: {patient_id}")
            cursor.close()
            conn.close()
            return False
        
        # Convert row to dict
        summary_data = dict(summary_result)
        
        # Get symptom logs
        cursor.execute("""
            SELECT * FROM symptom_logs 
            WHERE patient_about_id = %s 
            ORDER BY symptom_date DESC
        """, (patient_id,))
        symptom_rows = cursor.fetchall()
        
        # Convert rows to list of dicts
        symptom_data = [dict(row) for row in symptom_rows]
        
        # Store in patient_context table
        cursor.execute("""
            INSERT INTO patient_context (patient_id, summary_data, symptom_data, last_updated)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (patient_id) 
            DO UPDATE SET 
                summary_data = EXCLUDED.summary_data,
                symptom_data = EXCLUDED.symptom_data,
                last_updated = NOW()
        """, (patient_id, json.dumps(summary_data), json.dumps(symptom_data)))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Loaded context for patient ID: {patient_id}")
        return True
        
    except Exception as e:
        logger.exception(f"Error loading patient context: {str(e)}")
        return False

def get_patient_context(patient_id: str) -> Dict[str, Any]:
    """
    Get patient context data
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Dict[str, Any]: Patient context data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get context data
        cursor.execute("""
            SELECT summary_data, symptom_data, last_updated
            FROM patient_context
            WHERE patient_id = %s
        """, (patient_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            # Format last_updated
            last_updated = result['last_updated']
            if isinstance(last_updated, datetime.datetime):
                last_updated = last_updated.isoformat()
                
            return {
                'summary_data': result['summary_data'],
                'symptom_data': result['symptom_data'],
                'last_updated': last_updated
            }
        else:
            # If no context found, try loading it
            loaded = load_patient_context(patient_id)
            if loaded:
                # Try getting it again
                return get_patient_context(patient_id)
            
            logger.warning(f"No context found for patient ID: {patient_id}")
            return {'summary_data': {}, 'symptom_data': [], 'last_updated': None}
        
    except Exception as e:
        logger.exception(f"Error getting patient context: {str(e)}")
        return {'summary_data': {}, 'symptom_data': [], 'last_updated': None}

def get_patient_profile(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Get patient profile including both verified_patients and patients table data
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Optional[Dict[str, Any]]: Patient profile data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get verified patient data
        cursor.execute("""
            SELECT email, first_name, last_name, date_of_birth, gender, last_login
            FROM verified_patients
            WHERE patient_id = %s
        """, (patient_id,))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            cursor.close()
            conn.close()
            logger.warning(f"No verified patient found with ID: {patient_id}")
            return None
        
        # Convert to dict
        user_data = dict(user_data)
        
        # Get context data
        context = get_patient_context(patient_id)
        
        # Format date fields
        if user_data['date_of_birth'] and isinstance(user_data['date_of_birth'], datetime.date):
            user_data['date_of_birth'] = user_data['date_of_birth'].isoformat()
            
        if user_data['last_login'] and isinstance(user_data['last_login'], datetime.datetime):
            user_data['last_login'] = user_data['last_login'].isoformat()
        
        # Combine data
        profile = {
            **user_data,
            'patient_id': patient_id,
            'medical_data': context.get('summary_data', {})
        }
        
        cursor.close()
        conn.close()
        
        return profile
        
    except Exception as e:
        logger.exception(f"Error getting patient profile: {str(e)}")
        return None

def get_patient_symptoms(patient_id: str) -> List[Dict[str, Any]]:
    """
    Get patient symptom data
    
    Args:
        patient_id: Patient ID
        
    Returns:
        List[Dict[str, Any]]: List of symptom records
    """
    try:
        # Get patient context
        context = get_patient_context(patient_id)
        
        symptom_data = context.get('symptom_data', [])
        
        # If symptom_data is a string, try to parse it
        if isinstance(symptom_data, str):
            try:
                symptom_data = json.loads(symptom_data)
            except json.JSONDecodeError:
                symptom_data = []
        
        return symptom_data
        
    except Exception as e:
        logger.exception(f"Error getting patient symptoms: {str(e)}")
        return []