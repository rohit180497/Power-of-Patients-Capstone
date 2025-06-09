import os
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import json

print(os.getcwd())
# Import our custom modules
from patient_auth_sys import PatientAuthenticator
from app.patient_agent_old import ProfessionalPatientAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize our systems
authenticator = PatientAuthenticator()
patient_agent = ProfessionalPatientAgent()

# Store for active sessions (in production, use Redis or database)
active_sessions = {}

class PowerOfPatientsApp:
    """
    Main application class that integrates authentication and chat functionality
    """
    
    def __init__(self):
        self.app = app
        self.authenticator = authenticator
        self.patient_agent = patient_agent
        self.setup_routes()
        logger.info("Power of Patients Application initialized")
    
    async def initialize_connections(self):
        """Initialize database connections for both systems"""
        try:
            # Connect authenticator to database
            auth_connected = await self.authenticator.connect_to_database()
            if not auth_connected:
                logger.error("Failed to connect authenticator to database")
                return False
            
            # Connect patient agent to database
            agent_connected = await self.patient_agent.connect_to_database()
            if not agent_connected:
                logger.error("Failed to connect patient agent to database")
                return False
            
            logger.info("✅ All database connections established successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error initializing connections: {e}")
            return False
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        # Serve static files (HTML, CSS, JS)
        @app.route('/')
        def serve_login():
            return self.get_login_html()
        
        @app.route('/index.html')
        def serve_login_alt():
            return self.get_login_html()
        
        @app.route('/chat.html')
        def serve_chat():
            return self.get_chat_html()
        
        # API Routes
        @app.route('/api/authenticate', methods=['POST'])
        def authenticate():
            return asyncio.run(self.handle_authentication())
        
        @app.route('/api/chat', methods=['POST'])
        def chat():
            return asyncio.run(self.handle_chat())
        
        @app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "authenticator": bool(self.authenticator.db_connection),
                    "patient_agent": bool(self.patient_agent.db_connection)
                }
            })
    
    async def handle_authentication(self):
        """Handle patient authentication"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No data provided",
                    "message": "Please provide email and password"
                }), 400
            
            email = data.get('email', '').strip()
            password = data.get('password', '').strip()
            
            logger.info(f"Authentication attempt for email: {email}")
            
            # Authenticate user
            auth_result = await self.authenticator.authenticate_patient(email, password)
            
            if auth_result['success']:
                # Update last login
                # await self.authenticator.update_last_login(auth_result['patient_id'])
                
                # Store session info
                session_id = f"{auth_result['patient_id']}_{datetime.now().timestamp()}"
                active_sessions[session_id] = {
                    'patient_id': auth_result['patient_id'],
                    'email': auth_result['email'],
                    'login_time': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Authentication successful for patient: {auth_result['patient_id']}")
                
                return jsonify(auth_result)
            else:
                logger.warning(f"❌ Authentication failed for email: {email}")
                return jsonify(auth_result), 401
                
        except Exception as e:
            logger.exception(f"Error in authentication: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Authentication service error"
            }), 500
    
    async def handle_chat(self):
        """Handle chat messages with Sallie"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No data provided"
                }), 400
            
            patient_id = data.get('patient_id')
            message = data.get('message', '').strip()
            
            if not patient_id:
                return jsonify({
                    "success": False,
                    "error": "Patient ID required"
                }), 400
            
            logger.info(f"Chat request from patient {patient_id}: {message[:50]}...")
            
            # Handle initial connection (welcome message)
            if message == 'INITIAL_CONNECTION':
                # Process empty query to trigger welcome message
                result = await self.patient_agent.process_query('', patient_id)
            else:
                # Process regular chat message
                result = await self.patient_agent.process_query(message, patient_id)
            
            # Log the interaction
            if result['success']:
                logger.info(f"✅ Chat response sent to patient {patient_id} via {result['agent_used']}")
                if result.get('paraphrased_query'):
                    logger.info(f"🔄 Query paraphrased: '{message}' → '{result['paraphrased_query']}'")
            else:
                logger.error(f"❌ Chat error for patient {patient_id}: {result.get('error', 'Unknown error')}")
            
            return jsonify(result)
            
        except Exception as e:
            logger.exception(f"Error in chat handling: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "response": "I'm experiencing technical difficulties. Please try again in a moment."
            }), 500
    
    def get_login_html(self):
        """Return the login HTML page"""
        # In production, you'd serve this as a static file
        with open('templates/login.html', 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_chat_html(self):
        """Return the chat HTML page"""
        with open('templates/chat.html', 'r', encoding='utf-8') as f:
            return f.read()

# Create application instance
power_app = PowerOfPatientsApp()

# Startup function
async def startup():
    """Initialize the application"""
    logger.info("🚀 Starting Power of Patients Application...")
    
    success = await power_app.initialize_connections()
    if success:
        logger.info("✅ Application startup complete")
        return True
    else:
        logger.error("❌ Application startup failed")
        return False

# Development server runner
def run_development_server():
    """Run the development server"""
    print("=" * 60)
    print("🏥 POWER OF PATIENTS - HEALTHCARE ASSISTANT")
    print("=" * 60)
    print("Starting application server...")
    
    # Initialize connections
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    startup_success = loop.run_until_complete(startup())
    
    if startup_success:
        print("\n✅ Server initialized successfully!")
        print("\n📋 Available endpoints:")
        print("   • http://localhost:5000/          - Login page")
        print("   • http://localhost:5000/chat.html - Chat interface")
        print("   • http://localhost:5000/api/health - Health check")
        print("\n🔐 Authentication API:")
        print("   • POST /api/authenticate - Login with email/password")
        print("   • POST /api/chat        - Send message to Sallie")
        print("\n💡 Usage:")
        print("   1. Go to http://localhost:5000/")
        print("   2. Login with your patient credentials")
        print("   3. Chat with Sallie about your health!")
        print("\n" + "=" * 60)
        
        # Start Flask development server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Avoid double initialization
        )
    else:
        print("❌ Failed to start server. Check your database configuration.")
        print("\n🔧 Troubleshooting:")
        print("   • Verify database connection settings in .env file")
        print("   • Ensure patient_auth table exists with columns: email, password, patient_id")
        print("   • Check that patient_summary table exists for patient agent")

# Command line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test-auth":
            # Test authentication system
            async def test():
                await startup()
                
                print("\n🧪 Testing Authentication System...")
                test_email = input("Enter test email: ").strip()
                test_password = input("Enter test password: ").strip()
                
                result = await authenticator.authenticate_patient(test_email, test_password)
                
                if result['success']:
                    print(f"✅ Authentication successful!")
                    print(f"   Patient ID: {result['patient_id']}")
                    print(f"   Email: {result['email']}")
                    
                    # Test chat
                    print(f"\n🤖 Testing chat with patient {result['patient_id']}...")
                    chat_result = await patient_agent.process_query("Hello Sallie!", result['patient_id'])
                    
                    if chat_result['success']:
                        print(f"✅ Chat test successful!")
                        print(f"   Response: {chat_result['response'][:100]}...")
                    else:
                        print(f"❌ Chat test failed: {chat_result.get('error')}")
                        
                else:
                    print(f"❌ Authentication failed: {result['message']}")
            
            asyncio.run(test())
            
        elif command == "test-chat":
            # Test chat system
            async def test_chat():
                await startup()
                
                patient_id = input("Enter patient ID: ").strip()
                
                while True:
                    message = input(f"\n[{patient_id}] Your message (or 'quit'): ").strip()
                    if message.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    result = await patient_agent.process_query(message, patient_id)
                    
                    if result['success']:
                        print(f"\n🤖 Sallie: {result['response']}")
                        if result.get('paraphrased_query'):
                            print(f"🔄 Enhanced: '{message}' → '{result['paraphrased_query']}'")
                    else:
                        print(f"❌ Error: {result.get('error')}")
            
            asyncio.run(test_chat())
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  • python main_app.py              - Start web server")
            print("  • python main_app.py test-auth    - Test authentication")
            print("  • python main_app.py test-chat    - Test chat system")
    else:
        # Start web server
        run_development_server()