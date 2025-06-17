from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
import os
import uuid
from datetime import datetime, timedelta
import jwt
import bcrypt
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Construct connection string from environment variables
SUPABASE_CONNECTION_STRING = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')

# Create SQLAlchemy engine and session
engine = create_engine(SUPABASE_CONNECTION_STRING)
db_session = scoped_session(sessionmaker(bind=engine))

CORS(app)

class AuthService:
    """Authentication service with various login methods"""
    
    @staticmethod
    def generate_token(user_id, email, role):
        """Generate JWT token for authentication"""
        token_payload = {
            'user_id': str(user_id),
            'email': email,
            'role': role,
            'exp': datetime.now() + timedelta(days=1)
        }
        return jwt.encode(token_payload, app.config['SECRET_KEY'], algorithm='HS256')
    
    @staticmethod
    def verify_token(token):
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def hash_password(password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def check_password(stored_password, provided_password):
        """Check if provided password matches stored hash"""
        return bcrypt.checkpw(
            provided_password.encode('utf-8'), 
            stored_password.encode('utf-8')
        )

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    data = request.json
    
    # Validate input
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400
    
    try:
        # Check if user already exists
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": data['email']}
            )
            existing_user = result.fetchone()
            
            if existing_user:
                return jsonify({"error": "Email already registered"}), 409
        
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        
        # Hash password
        hashed_password = AuthService.hash_password(data['password'])
        
        # Insert new user
        with engine.connect() as connection:
            connection.execute(
                text("""
                    INSERT INTO users 
                    (id, email, password_hash, name, role, created_at) 
                    VALUES (:id, :email, :password_hash, :name, :role, :created_at)
                """),
                {
                    "id": user_id,
                    "email": data['email'],
                    "password_hash": hashed_password,
                    "name": data.get('name', ''),
                    "role": data.get('role', 'patient'),
                    "created_at": datetime.utcnow()
                }
            )
            connection.commit()
        
        # Generate authentication token
        token = AuthService.generate_token(
            user_id, 
            data['email'], 
            data.get('role', 'patient')
        )
        
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": user_id,
                "email": data['email'],
                "name": data.get('name', ''),
                "role": data.get('role', 'patient')
            }
        }), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.json
    
    # Validate input
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400
    
    try:
        # Find user
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": data['email']}
            )
            user = result.fetchone()
        
        # Verify credentials
        if user and AuthService.check_password(
            user.password_hash, 
            data['password']
        ):
            # Generate authentication token
            token = AuthService.generate_token(
                user.id, 
                user.email, 
                user.role
            )
            
            return jsonify({
                "message": "Login successful",
                "token": token,
                "user": {
                    "id": str(user.id),
                    "email": user.email,
                    "name": user.name,
                    "role": user.role
                }
            }), 200
        
        return jsonify({"error": "Invalid credentials"}), 401
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/verify-token', methods=['POST'])
def verify_token():
    """Token verification endpoint"""
    data = request.json
    token = data.get('token')
    
    if not token:
        return jsonify({"error": "Token is required"}), 400
    
    payload = AuthService.verify_token(token)
    
    if payload:
        return jsonify({
            "valid": True,
            "user": {
                "id": payload['user_id'],
                "email": payload['email'],
                "role": payload['role']
            }
        }), 200
    
    return jsonify({"valid": False, "error": "Invalid or expired token"}), 401

@app.route('/api/user/role-selection', methods=['POST'])
def select_user_role():
    """Endpoint for selecting or updating user role"""
    data = request.json
    token = data.get('token')
    role = data.get('role')
    
    if not token or not role:
        return jsonify({"error": "Token and role are required"}), 400
    
    payload = AuthService.verify_token(token)
    
    if not payload:
        return jsonify({"error": "Invalid token"}), 401
    
    try:
        # Update user role
        with engine.connect() as connection:
            connection.execute(
                text("UPDATE users SET role = :role WHERE id = :user_id"),
                {
                    "role": role,
                    "user_id": payload['user_id']
                }
            )
            connection.commit()
        
        return jsonify({
            "message": "Role updated successfully",
            "user": {
                "id": payload['user_id'],
                "email": payload['email'],
                "role": role
            }
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Print connection details for debugging (remove in production)
print(f"Connecting to database at {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}")

if __name__ == '__main__':
    app.run(debug=True)