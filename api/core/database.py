import psycopg2
import psycopg2.extras
import logging
from core.config import DB_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection"""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn

def initialize_db():
    """Create necessary database tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create verified_patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verified_patients (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255),
            google_id VARCHAR(255) UNIQUE,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            date_of_birth DATE,
            gender VARCHAR(50),
            is_verified BOOLEAN DEFAULT FALSE,
            verification_code VARCHAR(10),
            verification_expiry TIMESTAMP,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create patient_chat_sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_chat_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            patient_id VARCHAR(255) NOT NULL REFERENCES verified_patients(patient_id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create patient_chat_messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_chat_messages (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL REFERENCES patient_chat_sessions(session_id),
            message TEXT NOT NULL,
            is_patient BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create patient_context table for storing retrieved patient data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_context (
            id SERIAL PRIMARY KEY,
            patient_id VARCHAR(255) UNIQUE NOT NULL REFERENCES verified_patients(patient_id),
            summary_data JSONB,
            symptom_data JSONB,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.close()
    conn.close()
    logger.info("Database initialization complete")