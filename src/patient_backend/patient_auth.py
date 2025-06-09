from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
import uvicorn
import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
import psycopg2
import psycopg2.extras
import json
import uuid
import datetime
import smtplib
import random
import string
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import google.generativeai as genai
import jwt
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Power of Patient API", description="API for Power of Patient TBI management platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET") or os.urandom(24).hex()
JWT_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds
JWT_ALGORITHM = "HS256"

# Security scheme
security = HTTPBearer()

# Database connection info
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
    'verification_expiry_hours': 24
}

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database connection function
def get_db_connection():
    """Create a database connection"""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn

# Initialize database tables
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

# Initialize database on startup
initialize_db()

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_SAFETY_SETTINGS = {
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }
    logger.info("Gemini API configured")
else:
    logger.warning("Gemini API key not found. LLM features will be limited.")

# Helper functions
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

def generate_verification_code(length: int = 6) -> str:
    """Generate a random verification code"""
    return ''.join(random.choices(string.digits, k=length))

def create_jwt_token(patient_id: str) -> str:
    """Create a JWT token for a patient"""
    payload = {
        "sub": patient_id,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=JWT_EXPIRATION),
        "iat": datetime.datetime.now(datetime.timezone.utc)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt_token(token: str) -> Optional[str]:
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

def send_verification_email(email: str, first_name: str, verification_code: str) -> bool:
    """Send a verification email to the patient"""
    try:
        message = MIMEMultipart()
        message['From'] = EMAIL_CONFIG['from_email']
        message['To'] = email
        message['Subject'] = "Verify Your Power of Patient Account"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Hello {first_name},</h2>
            <p>Thank you for joining Power of Patient. Please use the following verification code to confirm your identity:</p>
            <div style="background-color: #f2f2f2; padding: 15px; text-align: center; font-size: 24px; letter-spacing: 5px;">
                <strong>{verification_code}</strong>
            </div>
            <p>This code will expire in 24 hours.</p>
            <p>If you did not request this verification, please ignore this email.</p>
            <p>Best regards,<br>The Power of Patient Team</p>
        </body>
        </html>
        """
        
        message.attach(MIMEText(body, 'html'))
        
        # Send the email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_user'], EMAIL_CONFIG['smtp_password'])
        server.send_message(message)
        server.quit()
        
        logger.info(f"Verification email sent to {email}")
        return True
        
    except Exception as e:
        logger.exception(f"Error sending verification email: {str(e)}")
        return False

async def parse_patient_input(input_text: str) -> tuple[str, str, str]:
    """
    Use LLM to parse potentially ambiguous patient input
    
    Args:
        input_text (str): Raw input text containing name and DOB information
        
    Returns:
        Tuple[str, str, str]: Parsed (first_name, last_name, dob)
    """
    if not GEMINI_API_KEY:
        # Fallback to basic parsing if no Gemini API key
        return basic_parse_input(input_text)
    
    try:
        # Initialize the model with high determinism
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config={
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 1,
                "max_output_tokens": 100,
            },
            safety_settings=GEMINI_SAFETY_SETTINGS
        )
        
        # Direct, concise prompt focused only on extraction and using explicit format
        prompt = f"""
Return ONLY a JSON object with first_name, last_name, and dob fields extracted from this text: "{input_text}"

The date should be formatted as YYYY-MM-DD.
Do not include any explanation, just return the JSON object.

For example:
- Input: "John Smith, born January 12, 1990"
- Output: {{"first_name": "John", "last_name": "Smith", "dob": "1990-01-12"}}

Return this exact JSON format and nothing else:
"""
        
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip()
        
        # Handle the case where the response includes a code block
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Try to parse the JSON response
        try:
            # Remove any non-JSON content at the beginning or end
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Extract the fields
                first_name = parsed_data.get('first_name', '')
                last_name = parsed_data.get('last_name', '')
                dob = parsed_data.get('dob', '')
                
                return first_name, last_name, dob
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {e} - Response: {response_text}")
        
        # If JSON parsing fails, fall back to basic parsing
        return basic_parse_input(input_text)
        
    except Exception as e:
        logger.exception(f"Error with Gemini parsing: {str(e)}")
        return basic_parse_input(input_text)

def basic_parse_input(input_text: str) -> tuple[str, str, str]:
    """
    Basic fallback method to parse patient input without LLM
    
    Args:
        input_text (str): Raw input text containing name and DOB information
        
    Returns:
        Tuple[str, str, str]: Basic parsing of (first_name, last_name, dob)
    """
    # Simple date extraction
    dob = None
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', input_text)
    if date_match:
        dob = date_match.group(0)
    
    # Simple name extraction
    first_name = ""
    last_name = ""
    
    # Try comma-separated last name, first name format
    comma_match = re.search(r'([A-Za-z]+),\s*([A-Za-z]+)', input_text)
    if comma_match:
        first_name = comma_match.group(2)
        last_name = comma_match.group(1)
    else:
        # Try standard first name, last name format
        name_match = re.search(r'([A-Za-z]+)\s+([A-Za-z]+)', input_text)
        if name_match:
            first_name = name_match.group(1)
            last_name = name_match.group(2)
    
    return first_name, last_name, dob

async def verify_patient_in_database(input_text: str) -> Optional[str]:
    """Verify if a patient exists in the database using the input text"""
    try:
        # Parse the input text to extract first name, last name, and DOB
        first_name, last_name, dob = await parse_patient_input(input_text)
        
        if not first_name or not last_name or not dob:
            logger.warning(f"Incomplete patient information parsed: {first_name}, {last_name}, {dob}")
            return None
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Try exact match first
        cursor.execute("""
            SELECT patient_id FROM patients 
            WHERE LOWER(first_name) = LOWER(%s) 
            AND LOWER(last_name) = LOWER(%s) 
            AND date_of_birth = %s
        """, (first_name, last_name, dob))
        
        result = cursor.fetchone()
        
        # If no exact match, try with swapped names
        if not result:
            cursor.execute("""
                SELECT patient_id FROM patients 
                WHERE LOWER(first_name) = LOWER(%s) 
                AND LOWER(last_name) = LOWER(%s) 
                AND date_of_birth = %s
            """, (last_name, first_name, dob))
            
            result = cursor.fetchone()
        
        # If still no match, try with LIKE
        if not result:
            cursor.execute("""
                SELECT patient_id FROM patients 
                WHERE (LOWER(first_name) LIKE LOWER(%s) OR LOWER(last_name) LIKE LOWER(%s))
                AND date_of_birth = %s
            """, (f"{first_name}%", f"{last_name}%", dob))
            
            result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return result['patient_id']
        
        logger.warning(f"No patient found with parsed data: {first_name}, {last_name}, {dob}")
        return None
        
    except Exception as e:
        logger.exception(f"Error verifying patient: {str(e)}")
        return None

async def verify_patient(first_name: str, last_name: str, dob: str) -> Optional[str]:
    """Verify if a patient exists in the database using structured inputs"""
    # Combine inputs into a single text string for processing
    input_text = f"{first_name} {last_name}, {dob}"
    return await verify_patient_in_database(input_text)

def create_or_update_verified_patient(patient_id: str, email: str, verification_data: Dict[str, Any]) -> bool:
    """Create or update a verified patient record"""
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
    """Verify a patient's verification code"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify code and check expiry
        cursor.execute("""
            SELECT verification_code, verification_expiry
            FROM verified_patients
            WHERE patient_id = %s AND is_verified = FALSE
        """, (patient_id,))
        
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            logger.warning(f"No pending verification found for patient ID: {patient_id}")
            return False
        
        stored_code, expiry = result
        
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
            cursor.close()
            
            # Load patient data into context
            load_patient_context(patient_id)
            
            conn.close()
            return True
        else:
            cursor.close()
            conn.close()
            logger.warning(f"Invalid or expired verification code for patient ID: {patient_id}")
            return False
        
    except Exception as e:
        logger.exception(f"Error verifying code: {str(e)}")
        return False

def load_patient_context(patient_id: str) -> bool:
    """Load patient context data from patient_summary and symptom_logs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get patient summary data
        cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
        summary_data = dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
        
        # Get symptom logs
        cursor.execute("""
            SELECT * FROM symptom_logs 
            WHERE patient_about_id = %s 
            ORDER BY symptom_date DESC
        """, (patient_id,))
        symptom_data = [dict(row) for row in cursor.fetchall()]
        
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
    """Get patient context data"""
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
            return {
                'summary_data': result['summary_data'],
                'symptom_data': result['symptom_data'],
                'last_updated': result['last_updated'].isoformat()
            }
        else:
            logger.warning(f"No context found for patient ID: {patient_id}")
            return {}
        
    except Exception as e:
        logger.exception(f"Error getting patient context: {str(e)}")
        return {}

def create_chat_session(patient_id: str) -> Optional[str]:
    """Create a new chat session for the patient"""
    try:
        session_id = str(uuid.uuid4())
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create session
        cursor.execute("""
            INSERT INTO patient_chat_sessions (session_id, patient_id)
            VALUES (%s, %s)
        """, (session_id, patient_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Created chat session {session_id} for patient ID: {patient_id}")
        return session_id
        
    except Exception as e:
        logger.exception(f"Error creating chat session: {str(e)}")
        return None

def store_chat_message(session_id: str, message: str, is_patient: bool) -> int:
    """Store a chat message"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Store message
        cursor.execute("""
            INSERT INTO patient_chat_messages (session_id, message, is_patient)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (session_id, message, is_patient))
        
        message_id = cursor.fetchone()[0]
        
        # Update session timestamp
        cursor.execute("""
            UPDATE patient_chat_sessions
            SET updated_at = NOW()
            WHERE session_id = %s
        """, (session_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return message_id
        
    except Exception as e:
        logger.exception(f"Error storing chat message: {str(e)}")
        return -1

def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get chat history for a session"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get messages
        cursor.execute("""
            SELECT id, message, is_patient, created_at
            FROM patient_chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (session_id, limit))
        
        messages = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return messages
        
    except Exception as e:
        logger.exception(f"Error getting chat history: {str(e)}")
        return []

def get_patient_chat_sessions(patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get a patient's chat sessions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get sessions
        cursor.execute("""
            SELECT session_id, created_at, updated_at
            FROM patient_chat_sessions
            WHERE patient_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
        """, (patient_id, limit))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        
        # Get message count for each session
        for session in sessions:
            cursor.execute("""
                SELECT COUNT(*) FROM patient_chat_messages
                WHERE session_id = %s
            """, (session['session_id'],))
            
            session['message_count'] = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return sessions
        
    except Exception as e:
        logger.exception(f"Error getting patient chat sessions: {str(e)}")
        return []

def generate_bot_response(message: str, context: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    """
    Generate a bot response based on the patient's message and context
    
    In a real implementation, this would use an LLM or other AI system.
    This is a simple placeholder implementation.
    """
    # Get basic patient info
    patient_name = context.get('summary_data', {}).get('first_name', 'Patient')
    
    # Count symptoms
    symptom_count = len(context.get('symptom_data', []))
    
    # Simple keyword-based responses
    if 'symptom' in message.lower():
        return f"Hi {patient_name}, I can see you have {symptom_count} symptom records in our system. How can I help you with your symptoms today?"
    
    if 'history' in message.lower():
        return f"I can see your medical history, {patient_name}. Is there anything specific about your history you'd like to discuss?"
    
    if 'help' in message.lower():
        return f"I'm here to help you manage your TBI recovery, {patient_name}. I can answer questions about your symptoms, provide resources, or connect you with your healthcare provider."
    
    # Default response
    return f"Thank you for your message, {patient_name}. As your TBI management assistant, I'm here to help you track and understand your symptoms. How are you feeling today?"

# Define Pydantic models for request/response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

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
    google_email: Optional[str] = None

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    dob: str
    gender: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class VerifyCodeRequest(BaseModel):
    patient_id: str
    verification_code: str

class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatMessageResponse(BaseModel):
    id: int
    message: str
    is_patient: bool
    session_id: str
    created_at: str

class ChatSession(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int

# API Routes

# Google OAuth endpoints
@app.post("/api/auth/google", response_model=Token)
async def google_auth(request: GoogleAuthRequest):
    """Handle Google OAuth authentication"""
    # In a real implementation, you would exchange the code for an access token
    # and then get the user info from Google
    # For this example, we'll simulate this process
    
    try:
        # Simulated Google user data
        google_id = "google_" + str(uuid.uuid4())
        google_email = f"user_{google_id[:8]}@gmail.com"
        google_first_name = "Google"
        google_last_name = "User"
        
        # Check if user exists
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Check if user exists by Google ID or email
        cursor.execute("""
            SELECT patient_id, is_verified 
            FROM verified_patients 
            WHERE google_id = %s OR email = %s
        """, (google_id, google_email))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            # Existing user
            patient_id = user['patient_id']
            is_verified = user['is_verified']
            
            if is_verified:
                # Generate token
                token = create_jwt_token(patient_id)
                return {"access_token": token, "token_type": "bearer"}
            else:
                # User needs to complete verification
                # In a real API, you might return a different response type
                # or set status code to indicate further action needed
                raise HTTPException(
                    status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                    detail={
                        "message": "Account not verified",
                        "patient_id": patient_id
                    }
                )
        else:
            # New user - Google OAuth flow not fully implemented in this example
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Google OAuth flow not fully implemented in this example API"
            )
    
    except Exception as e:
        logger.exception(f"Error in Google auth: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error"
        )

# Registration endpoints
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(request: RegisterRequest):
    """Register a new user with email and password"""
    try:
        # Verify patient in database
        patient_id = await verify_patient(request.first_name, request.last_name, request.dob)
        
        if not patient_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found in our records. Please check your information."
            )
        
        # Hash the password
        password_hash = hash_password(request.password)
        
        # Store patient info
        verification_data = {
            'patient_info': {
                'first_name': request.first_name,
                'last_name': request.last_name,
                'date_of_birth': request.dob,
                'gender': request.gender
            },
            'password_hash': password_hash
        }
        
        # Create or update verified patient
        result = create_or_update_verified_patient(patient_id, request.email, verification_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create account. Please try again."
            )
        
        return {"message": "Registration successful. Please check your email for a verification code.", "patient_id": patient_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration error"
        )

@app.post("/api/auth/complete-registration", status_code=status.HTTP_201_CREATED)
async def complete_google_registration(request: CompleteRegistrationRequest):
    """Complete registration for a Google OAuth user"""
    try:
        # Verify patient in database
        patient_id = await verify_patient(request.first_name, request.last_name, request.dob)
        
        if not patient_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found in our records. Please check your information."
            )
        
        # Store patient info
        verification_data = {
            'patient_info': {
                'first_name': request.first_name,
                'last_name': request.last_name,
                'date_of_birth': request.dob,
                'gender': request.gender
            },
            'google_id': request.google_id
        }
        
        # Create or update verified patient
        result = create_or_update_verified_patient(patient_id, request.google_email, verification_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to complete registration. Please try again."
            )
        
        # Load patient context
        load_patient_context(patient_id)
        
        # Generate JWT token
        token = create_jwt_token(patient_id)
        
        return {"access_token": token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in completing registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration error"
        )

# Verification endpoints
@app.post("/api/auth/verify-patient")
async def verify_patient_endpoint(request: VerificationRequest):
    """Verify patient identity and send verification code"""
    try:
        patient_id = None
        
        # Option 1: Separate fields
        if request.first_name and request.last_name and request.dob:
            patient_id = await verify_patient(request.first_name, request.last_name, request.dob)
        
        # Option 2: Flexible text input
        elif request.patient_info:
            patient_id = await verify_patient_in_database(request.patient_info)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide either structured fields (first_name, last_name, dob) or patient_info."
            )
        
        if not patient_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found. Please check your information."
            )
        
        # Extract name components for email
        if request.patient_info and not (request.first_name and request.last_name):
            # Extract name from patient_info for verification email
            first_name, last_name, _ = await parse_patient_input(request.patient_info)
        else:
            first_name = request.first_name or ""
            last_name = request.last_name or ""
        
        # Store patient info
        verification_data = {
            'patient_info': {
                'first_name': first_name,
                'last_name': last_name,
                'date_of_birth': request.dob
            }
        }
        
        # Create or update verified patient and send verification email
        result = create_or_update_verified_patient(patient_id, request.email, verification_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification email. Please try again."
            )
        
        return {"message": "Verification email sent. Please check your email for a verification code.", "patient_id": patient_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in patient verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification error"
        )

@app.post("/api/auth/verify-code", response_model=Token)
async def verify_code_endpoint(request: VerifyCodeRequest):
    """Verify a verification code and return a JWT token"""
    try:
        result = verify_code(request.patient_id, request.verification_code)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification code."
            )
        
        # Generate JWT token
        token = create_jwt_token(request.patient_id)
        
        return {"access_token": token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error verifying code: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification error"
        )

# Login endpoint
@app.post("/api/auth/login", response_model=Token)
async def login(request: LoginRequest):
    """Log in with email and password"""
    try:
        # Find user by email
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute("""
            SELECT patient_id, password_hash, is_verified
            FROM verified_patients
            WHERE email = %s
        """, (request.email,))
        
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        patient_id, password_hash, is_verified = user
        
        # Verify password
        if not verify_password(password_hash, request.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if verified
        if not is_verified:
            # Send new verification code
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT first_name FROM verified_patients WHERE patient_id = %s", (patient_id,))
            first_name = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            verification_data = {
                'patient_info': {
                    'first_name': first_name
                }
            }
            
            create_or_update_verified_patient(patient_id, request.email, verification_data)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "message": "Account not verified",
                    "patient_id": patient_id
                }
            )
        
        # Update last login
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE verified_patients SET last_login = NOW() WHERE patient_id = %s", (patient_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        # Generate JWT token
        token = create_jwt_token(patient_id)
        
        return {"access_token": token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login error"
        )

# Patient data endpoints
@app.get("/api/patient/profile")
async def get_patient_profile(patient_id: str = Depends(verify_token)):
    """Get patient profile data"""
    try:
        # Get patient context
        context = get_patient_context(patient_id)
        
        if not context or not context.get('summary_data'):
            # If context not found, try to load it
            load_patient_context(patient_id)
            context = get_patient_context(patient_id)
        
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient data not found"
            )
        
        # Get verified patient data
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute("""
            SELECT email, first_name, last_name, date_of_birth, gender, last_login
            FROM verified_patients
            WHERE patient_id = %s
        """, (patient_id,))
        
        user_data = dict(cursor.fetchone()) if cursor.rowcount > 0 else {}
        cursor.close()
        conn.close()
        
        # Combine data
        profile = {
            **user_data,
            'patient_id': patient_id,
            'medical_data': context.get('summary_data', {})
        }
        
        # Remove sensitive or redundant fields
        if 'password_hash' in profile:
            del profile['password_hash']
        
        return profile
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting patient profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving patient data"
        )

@app.get("/api/patient/symptoms")
async def get_patient_symptoms(patient_id: str = Depends(verify_token)):
    """Get patient symptom data"""
    try:
        # Get patient context
        context = get_patient_context(patient_id)
        
        if not context or not context.get('symptom_data'):
            # If context not found, try to load it
            load_patient_context(patient_id)
            context = get_patient_context(patient_id)
        
        symptom_data = context.get('symptom_data', [])
        
        return {"symptoms": symptom_data}
    
    except Exception as e:
        logger.exception(f"Error getting patient symptoms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving symptom data"
        )

# Chat endpoints
@app.get("/api/chat/sessions")
async def get_chat_sessions(patient_id: str = Depends(verify_token)):
    """Get a patient's chat sessions"""
    try:
        sessions = get_patient_chat_sessions(patient_id)
        return {"sessions": sessions}
    
    except Exception as e:
        logger.exception(f"Error getting chat sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving chat sessions"
        )

@app.post("/api/chat/sessions")
async def create_new_chat_session(patient_id: str = Depends(verify_token)):
    """Create a new chat session"""
    try:
        session_id = create_chat_session(patient_id)
        
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
        
        return {
            "session_id": session_id,
            "created_at": datetime.datetime.now().isoformat(),
            "message": "Chat session created successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating chat session"
        )

@app.get("/api/chat/history/{session_id}")
async def get_chat_session_history(session_id: str, patient_id: str = Depends(verify_token)):
    """Get chat history for a session"""
    try:
        # Verify that the session belongs to the patient
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT patient_id FROM patient_chat_sessions
            WHERE session_id = %s
        """, (session_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        session_patient_id = result[0]
        
        if session_patient_id != patient_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this chat session"
            )
        
        # Get chat history
        messages = get_chat_history(session_id)
        
        # Format messages for response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": msg["id"],
                "message": msg["message"],
                "is_patient": msg["is_patient"],
                "created_at": msg["created_at"].isoformat() if isinstance(msg["created_at"], datetime.datetime) else msg["created_at"]
            })
        
        return {"messages": formatted_messages}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving chat history"
        )

@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest, patient_id: str = Depends(verify_token)):
    """Send a chat message and get a response"""
    try:
        session_id = request.session_id
        
        # If no session ID provided, create a new session
        if not session_id:
            session_id = create_chat_session(patient_id)
            
            if not session_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create chat session"
                )
        else:
            # Verify that the session belongs to the patient
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT patient_id FROM patient_chat_sessions
                WHERE session_id = %s
            """, (session_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found"
                )
            
            session_patient_id = result[0]
            
            if session_patient_id != patient_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have access to this chat session"
                )
        
        # Store the patient's message
        message_id = store_chat_message(session_id, request.message, True)
        
        # Get patient context
        context = get_patient_context(patient_id)
        
        # Get chat history
        history = get_chat_history(session_id)
        
        # Generate a bot response
        bot_response = generate_bot_response(request.message, context, history)
        
        # Store the bot's response
        bot_message_id = store_chat_message(session_id, bot_response, False)
        
        # Get the timestamp
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT created_at FROM patient_chat_messages WHERE id = %s", (bot_message_id,))
        created_at = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Format the response
        return {
            "id": bot_message_id,
            "message": bot_response,
            "is_patient": False,
            "session_id": session_id,
            "created_at": created_at.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error sending chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat message"
        )

# Health check
@app.get("/health")
def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)