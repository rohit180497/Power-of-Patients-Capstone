import os
from flask import Flask, request, redirect, url_for, render_template, session, jsonify, flash
from flask_oauthlib.client import OAuth
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import uuid
import hashlib
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
import secrets
import string
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16)
oauth = OAuth(app)

# Configure Google OAuth
google = oauth.remote_app(
    'google',
    consumer_key=os.getenv("GOOGLE_CLIENT_ID"),
    consumer_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    request_token_params={
        'scope': 'email profile'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

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
    'from_email': os.getenv("FROM_EMAIL")
}

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET") or secrets.token_hex(32)
JWT_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn

# Create necessary tables if they don't exist
def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255),
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        date_of_birth DATE,
        is_verified BOOLEAN DEFAULT FALSE,
        verification_token VARCHAR(255),
        google_id VARCHAR(255) UNIQUE,
        patient_id VARCHAR(255) UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create session table for chat history
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        session_id VARCHAR(255) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create messages table for storing chat messages
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) REFERENCES chat_sessions(session_id),
        user_id INTEGER REFERENCES users(id),
        message TEXT NOT NULL,
        is_bot BOOLEAN NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.close()
    conn.close()

# Initialize database on startup
initialize_db()

# Helper functions
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwd_hash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwd_hash = hashlib.sha256(pwd_hash).hexdigest()
    return (salt + pwd_hash).decode('ascii')

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_pwd_hash = stored_password[64:]
    pwd_hash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwd_hash = hashlib.sha256(pwd_hash).hexdigest()
    return pwd_hash == stored_pwd_hash

def generate_verification_token():
    """Generate a token for email verification"""
    return secrets.token_urlsafe(32)

def send_verification_email(email, token):
    """Send verification email to user"""
    verification_url = url_for('verify_email', token=token, _external=True)
    
    message = MIMEMultipart()
    message['From'] = EMAIL_CONFIG['from_email']
    message['To'] = email
    message['Subject'] = "Verify Your Power of Patient Account"
    
    body = f"""
    <html>
    <body>
        <h2>Welcome to Power of Patient!</h2>
        <p>Thank you for registering. Please click the link below to verify your email address:</p>
        <p><a href="{verification_url}">Verify Email</a></p>
        <p>If you did not sign up for an account, please ignore this email.</p>
    </body>
    </html>
    """
    
    message.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_user'], EMAIL_CONFIG['smtp_password'])
        server.send_message(message)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def generate_jwt_token(user_id):
    """Generate a JWT token for a user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.now() + datetime.timedelta(seconds=JWT_EXPIRATION)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token):
    """Verify a JWT token and return user_id if valid"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

def verify_patient_in_database(first_name, last_name, dob):
    """Verify if a patient exists in the patient_summary table"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Format date of birth for database query
    try:
        # Try to parse the DOB in various formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
        parsed_dob = None
        
        for fmt in formats:
            try:
                parsed_dob = datetime.datetime.strptime(dob, fmt).strftime('%Y-%m-%d')
                break
            except ValueError:
                continue
        
        if not parsed_dob:
            return None
        
        # Query to find the patient in the database
        cursor.execute("""
            SELECT patient_id FROM patients 
            WHERE LOWER(first_name) = LOWER(%s) 
            AND LOWER(last_name) = LOWER(%s) 
            AND date_of_birth = %s
        """, (first_name, last_name, parsed_dob))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return result['patient_id']
        return None
        
    except Exception as e:
        print(f"Error verifying patient: {e}")
        cursor.close()
        conn.close()
        return None

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('token') or request.headers.get('Authorization')
        
        if not token:
            return redirect(url_for('login'))
        
        # Remove Bearer prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
            
        user_id = verify_jwt_token(token)
        if not user_id:
            return redirect(url_for('login'))
            
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and verify_password(user['password_hash'], password):
            if not user['is_verified']:
                flash('Please verify your email before logging in.', 'warning')
                return redirect(url_for('login'))
                
            token = generate_jwt_token(user['id'])
            response = redirect(url_for('dashboard'))
            response.set_cookie('token', token, httponly=True, max_age=JWT_EXPIRATION)
            return response
        else:
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        dob = request.form.get('date_of_birth')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Verify if patient exists in our database
        patient_id = verify_patient_in_database(first_name, last_name, dob)
        
        # Generate verification token
        verification_token = generate_verification_token()
        
        # Hash the password
        password_hash = hash_password(password)
        
        # Insert the new user
        cursor.execute("""
            INSERT INTO users (email, password_hash, first_name, last_name, date_of_birth, verification_token, patient_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (email, password_hash, first_name, last_name, dob, verification_token, patient_id))
        
        user_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        # Send verification email
        send_verification_email(email, verification_token)
        
        flash('Registration successful! Please check your email to verify your account.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/verify-email/<token>')
def verify_email(token):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE verification_token = %s", (token,))
    user = cursor.fetchone()
    
    if user:
        cursor.execute("UPDATE users SET is_verified = TRUE, verification_token = NULL WHERE id = %s", (user[0],))
        conn.commit()
        flash('Email verified successfully! You can now log in.', 'success')
    else:
        flash('Invalid verification token.', 'danger')
    
    cursor.close()
    conn.close()
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    response = redirect(url_for('index'))
    response.delete_cookie('token')
    return response

@app.route('/google-login')
def google_login():
    return google.authorize(callback=url_for('google_authorized', _external=True))

@app.route('/google-authorized')
def google_authorized():
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        return 'Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        )
    
    session['google_token'] = (resp['access_token'], '')
    me = google.get('userinfo')
    google_id = me.data['id']
    email = me.data['email']
    first_name = me.data.get('given_name', '')
    last_name = me.data.get('family_name', '')
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE google_id = %s OR email = %s", (google_id, email))
    user = cursor.fetchone()
    
    if user:
        # Update Google ID if necessary
        if not user['google_id']:
            cursor.execute("UPDATE users SET google_id = %s WHERE id = %s", (google_id, user['id']))
            conn.commit()
    else:
        # This is a new user - we need to collect more info
        session['temp_google_id'] = google_id
        session['temp_email'] = email
        session['temp_first_name'] = first_name
        session['temp_last_name'] = last_name
        
        cursor.close()
        conn.close()
        return redirect(url_for('complete_google_signup'))
    
    # Generate token and log in
    token = generate_jwt_token(user['id'])
    response = redirect(url_for('dashboard'))
    response.set_cookie('token', token, httponly=True, max_age=JWT_EXPIRATION)
    
    cursor.close()
    conn.close()
    return response

@app.route('/complete-google-signup', methods=['GET', 'POST'])
def complete_google_signup():
    if 'temp_google_id' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        dob = request.form.get('date_of_birth')
        first_name = request.form.get('first_name', session['temp_first_name'])
        last_name = request.form.get('last_name', session['temp_last_name'])
        
        # Verify if patient exists in our database
        patient_id = verify_patient_in_database(first_name, last_name, dob)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert the new user with Google authentication
        cursor.execute("""
            INSERT INTO users (email, first_name, last_name, date_of_birth, google_id, is_verified, patient_id)
            VALUES (%s, %s, %s, %s, %s, TRUE, %s)
            RETURNING id
        """, (session['temp_email'], first_name, last_name, dob, session['temp_google_id'], patient_id))
        
        user_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        # Clean up session
        for key in ['temp_google_id', 'temp_email', 'temp_first_name', 'temp_last_name']:
            session.pop(key, None)
        
        # Generate token and log in
        token = generate_jwt_token(user_id)
        response = redirect(url_for('dashboard'))
        response.set_cookie('token', token, httponly=True, max_age=JWT_EXPIRATION)
        return response
        
    return render_template('complete_google_signup.html', 
                          first_name=session.get('temp_first_name', ''),
                          last_name=session.get('temp_last_name', ''))

@app.route('/dashboard')
@login_required
def dashboard():
    token = request.cookies.get('token')
    user_id = verify_jwt_token(token)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get user details
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    # Get patient details if they're verified
    patient_data = None
    symptom_history = []
    
    if user and user['patient_id']:
        # Get patient summary
        cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (user['patient_id'],))
        patient_data = cursor.fetchone()
        
        # Get symptom history
        cursor.execute("""
            SELECT * FROM symptom_reference 
            WHERE patient_id = %s 
            ORDER BY id DESC
        """, (user['patient_id'],))
        symptom_history = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('dashboard.html', user=user, patient=patient_data, symptoms=symptom_history)

@app.route('/patient-verification', methods=['GET', 'POST'])
@login_required
def patient_verification():
    token = request.cookies.get('token')
    user_id = verify_jwt_token(token)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get user details
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        dob = request.form.get('date_of_birth')
        
        # Verify if patient exists in our database
        patient_id = verify_patient_in_database(first_name, last_name, dob)
        
        if patient_id:
            # Update user with patient_id
            cursor.execute("UPDATE users SET patient_id = %s WHERE id = %s", (patient_id, user_id))
            conn.commit()
            flash('Patient verification successful!', 'success')
            
            # Send verification email
            send_verification_email(user['email'], f"verify-patient-{patient_id}")
            flash('A verification email has been sent to confirm your identity.', 'info')
            
            cursor.close()
            conn.close()
            return redirect(url_for('dashboard'))
        else:
            flash('Patient not found in our records. Please check your information.', 'danger')
    
    cursor.close()
    conn.close()
    return render_template('patient_verification.html')

@app.route('/chat')
@login_required
def chat():
    token = request.cookies.get('token')
    user_id = verify_jwt_token(token)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get user details
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    # Check if user is verified and has a patient_id
    if not user['patient_id']:
        cursor.close()
        conn.close()
        flash('Please complete patient verification first.', 'warning')
        return redirect(url_for('patient_verification'))
    
    # Create or get a chat session
    cursor.execute("""
        SELECT * FROM chat_sessions 
        WHERE user_id = %s 
        ORDER BY updated_at DESC 
        LIMIT 1
    """, (user_id,))
    
    session = cursor.fetchone()
    
    if not session:
        session_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO chat_sessions (user_id, session_id)
            VALUES (%s, %s)
            RETURNING id
        """, (user_id, session_id))
        
        conn.commit()
        
        # Get the newly created session
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,))
        session = cursor.fetchone()
    
    # Get chat history
    cursor.execute("""
        SELECT * FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
    """, (session['session_id'],))
    
    messages = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('chat.html', user=user, session=session, messages=messages)

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    token = request.cookies.get('token')
    user_id = verify_jwt_token(token)
    
    data = request.json
    message = data.get('message')
    session_id = data.get('session_id')
    
    if not message or not session_id:
        return jsonify({'error': 'Missing message or session_id'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Get user details
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    # Store user message
    cursor.execute("""
        INSERT INTO chat_messages (session_id, user_id, message, is_bot)
        VALUES (%s, %s, %s, FALSE)
        RETURNING id
    """, (session_id, user_id, message))
    
    conn.commit()
    
    # Get the patient data for context
    cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (user['patient_id'],))
    patient = cursor.fetchone()
    
    # Get symptom history for context
    cursor.execute("""
        SELECT * FROM symptom_reference 
        WHERE patient_id = %s 
        ORDER BY id DESC 
        LIMIT 10
    """, (user['patient_id'],))
    symptoms = cursor.fetchall()
    
    # This is where you would call your AI agent/chatbot API
    # For now, we'll just simulate a response
    
    # Process the message and generate a response
    # This will be replaced with your actual PandasAgent code
    bot_response = process_patient_message(message, user, patient, symptoms)
    
    # Store bot response
    cursor.execute("""
        INSERT INTO chat_messages (session_id, user_id, message, is_bot)
        VALUES (%s, %s, %s, TRUE)
        RETURNING id, message, created_at
    """, (session_id, user_id, bot_response))
    
    response_data = cursor.fetchone()
    
    # Update session last activity time
    cursor.execute("UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s", (session_id,))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({
        'id': response_data['id'],
        'message': response_data['message'],
        'timestamp': response_data['created_at'].isoformat()
    })

def process_patient_message(message, user, patient, symptoms):
    """
    Process a message from a patient and generate a response.
    This is a placeholder and will be replaced with the actual agent code.
    """
    # Simply echo the message for now - this will be replaced with actual agent logic
    if "symptom" in message.lower():
        return f"Hi {user['first_name']}, I can see you have {len(symptoms)} symptom records in our system. Your most recent symptoms include: {', '.join([s['category'] for s in symptoms[:3]])}. How can I help you manage these symptoms today?"
    
    if "history" in message.lower():
        return f"I can see from your history that you've had a TBI incident recorded on {patient['tbi_incident_date']}. You've been tracking your symptoms with us for a while. Is there anything specific about your history you'd like to discuss?"
    
    if "help" in message.lower():
        return "I can help you with tracking your symptoms, understanding your TBI history, providing resources for recovery, or connecting you with your healthcare provider. What would you like help with today?"
    
    return f"Thank you for your message, {user['first_name']}. As your TBI management assistant, I'm here to help you track and understand your symptoms. How are you feeling today?"

if __name__ == '__main__':
    app.run(debug=True)