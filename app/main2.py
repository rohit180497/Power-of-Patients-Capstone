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
from researcher_auth import ResearcherAuthenticator
from patient_agent import ProfessionalPatientAgent
from researcheragent import ProfessionalResearcherAgent

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
patient_authenticator = PatientAuthenticator()
researcher_authenticator = ResearcherAuthenticator()
patient_agent = ProfessionalPatientAgent()
researcher_agent = ProfessionalResearcherAgent()

# Store for active sessions (in production, use Redis or database)
active_sessions = {}

class PowerOfPatientsApp:
    """
    Main application class that integrates authentication and chat functionality
    for both patients and researchers
    """
    
    def __init__(self):
        self.app = app
        self.patient_authenticator = patient_authenticator
        self.researcher_authenticator = researcher_authenticator
        self.patient_agent = patient_agent
        self.researcher_agent = researcher_agent
        self.setup_routes()
        logger.info("Power of Patients Application initialized with dual user support")
    
    async def initialize_connections(self):
        """Initialize database connections for all systems"""
        try:
            # Connect patient authenticator to database
            patient_auth_connected = await self.patient_authenticator.connect_to_database()
            if not patient_auth_connected:
                logger.error("Failed to connect patient authenticator to database")
                return False
            
            # Connect researcher authenticator to database
            researcher_auth_connected = await self.researcher_authenticator.connect_to_database()
            if not researcher_auth_connected:
                logger.error("Failed to connect researcher authenticator to database")
                return False
            
            # Connect patient agent to database
            patient_agent_connected = await self.patient_agent.connect_to_database()
            if not patient_agent_connected:
                logger.error("Failed to connect patient agent to database")
                return False
            
            # Connect researcher agent to database (includes PandasAgent)
            researcher_agent_connected = await self.researcher_agent.connect_to_database()
            if not researcher_agent_connected:
                logger.error("Failed to connect researcher agent to database")
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
        
        # Add route for researcher chat
        @app.route('/researcher_chat.html')
        def serve_researcher_chat():
            return self.get_researcher_chat_html()
        
        # Static file serving
        @app.route('/static/<path:filename>')
        def serve_static(filename):
            return send_from_directory('static', filename)
        
        # API Routes
        @app.route('/api/authenticate', methods=['POST'])
        def authenticate():
            return asyncio.run(self.handle_authentication())
        
        @app.route('/api/chat', methods=['POST'])
        def chat():
            return asyncio.run(self.handle_chat())
        
        # Additional researcher-specific routes
        @app.route('/api/researcher/examples', methods=['GET'])
        def get_research_examples():
            """Get example queries for researchers"""
            try:
                examples = self.researcher_agent.get_available_analyses()
                return jsonify({
                    "success": True,
                    "examples": examples
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "patient_authenticator": bool(self.patient_authenticator.db_connection),
                    "researcher_authenticator": bool(self.researcher_authenticator.db_connection),
                    "patient_agent": bool(self.patient_agent.db_connection),
                    "researcher_agent": bool(self.researcher_agent.db_connection),
                    "pandas_agent": hasattr(self.researcher_agent, 'pandas_agent') and bool(self.researcher_agent.pandas_agent.dataframes)
                }
            })
        
        @app.route('/api/logout', methods=['POST'])
        def api_logout():
            return asyncio.run(self.handle_logout())

        # And add this method to your PowerOfPatientsApp class:

        async def handle_logout(self):
            """Handle user logout and cleanup connections"""
            try:
                # Get session ID from request if available
                data = request.get_json() or {}
                user_id = data.get('patient_id') or data.get('user_id')
                user_type = data.get('user_type', 'unknown')
                
                logger.info(f"Logout request from {user_type} {user_id}")
                
                # Remove from active sessions if being tracked
                for session_id, session_data in list(active_sessions.items()):
                    if session_data.get('user_id') == user_id:
                        del active_sessions[session_id]
                        logger.info(f"Removed session for {user_id}")
                        break
                try:
                    if user_type == 'researcher':
                        # Clear researcher agent memory if it has the method
                        if hasattr(self.researcher_agent, 'clear_conversation_history'):
                            await self.researcher_agent.clear_conversation_history(user_id)
                        else:
                            logger.info("Researcher agent doesn't have clear_conversation_history method")
                    else:
                        # Clear patient agent memory
                        await self.patient_agent.clear_conversation_history(user_id)
                        
                    logger.info(f"✅ Cleared conversation memory for {user_type} {user_id}")
                    
                except Exception as memory_error:
                    logger.error(f"Error clearing memory for {user_id}: {memory_error}")
                    # Don't fail the logout if memory clearing fails
                
                return jsonify({
                    "success": True,
                    "message": "Logged out successfully"
                })
                # Don't close connections here - we'll manage them with ensure_connection
                # when the user tries to log in again
                    
            except Exception as e:
                logger.exception(f"Error in logout: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "message": "Error during logout"
                }), 500
    
    async def handle_authentication(self):
        """Handle authentication - check patient first, then researcher"""
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
            
            # First, try patient authentication
            patient_result = await self.patient_authenticator.authenticate_patient(email, password)
            
            if patient_result['success']:
                # Patient authentication successful
                session_id = f"{patient_result['patient_id']}_{datetime.now().timestamp()}"
                active_sessions[session_id] = {
                    'user_id': patient_result['patient_id'],
                    'email': patient_result['email'],
                    'user_type': 'patient',
                    'login_time': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Patient authentication successful: {patient_result['patient_id']}")
                
                # Add user_type and redirect URL to response
                patient_result['user_type'] = 'patient'
                patient_result['redirect_url'] = '/chat.html'
                return jsonify(patient_result)
            
            # If patient auth failed, try researcher authentication
            logger.info(f"Patient auth failed, trying researcher authentication for: {email}")
            print(email, password)
            researcher_result = await self.researcher_authenticator.authenticate_researcher(email, password)
            
            if researcher_result['success']:
                # Researcher authentication successful
                session_id = f"{researcher_result['researcher_id']}_{datetime.now().timestamp()}"
                active_sessions[session_id] = {
                    'user_id': researcher_result['researcher_id'],
                    'email': researcher_result['email'],
                    'user_type': 'researcher',
                    'login_time': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Researcher authentication successful: {researcher_result['researcher_id']}")
                
                # Map researcher_id to patient_id for frontend compatibility
                researcher_result['patient_id'] = researcher_result['researcher_id']
                researcher_result['user_type'] = 'researcher'
                researcher_result['redirect_url'] = '/researcher_chat.html'
                return jsonify(researcher_result)
            
            # Both authentications failed
            logger.warning(f"❌ All authentication attempts failed for email: {email}")
            return jsonify({
                "success": False,
                "error": "Invalid credentials",
                "message": "Invalid email or password. Please check your credentials and try again."
            }), 401
                
        except Exception as e:
            logger.exception(f"Error in authentication: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Authentication service error"
            }), 500
    
    async def handle_chat(self):
        """Handle chat messages - route to appropriate agent based on user type"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No data provided"
                }), 400
            
            user_id = data.get('patient_id')  # Frontend still sends as patient_id
            message = data.get('message', '').strip()
            user_type = data.get('user_type', 'patient')  # Get user type from request
            
            if not user_id:
                return jsonify({
                    "success": False,
                    "error": "User ID required"
                }), 400
            
            # Handle initial connection for welcome message - FOR BOTH USER TYPES
            if message == 'INITIAL_CONNECTION':
                logger.info(f"Initial connection from {user_type} {user_id}")
                message = ''  # Reset to empty string for both user types
            else:
                logger.info(f"Chat request from {user_type} {user_id}: {message[:50]}...")
            
            # Route to appropriate agent based on user type
            if user_type == 'researcher':
                # Handle researcher queries
                result = await self.researcher_agent.process_query(message, user_id)
                
                if result['success']:
                    logger.info(f"✅ Researcher query processed for {user_id}")
                    
                    # Format response for frontend compatibility
                    response_data = {
                        "success": True,
                        "patient_id": user_id,  # Keep as patient_id for frontend
                        "patient_name": user_id,  # Use email as name for researchers
                        "query": message,
                        "intent_classified": result.get('intent', 'data_analysis'),
                        "agent_used": f"Researcher/{result.get('metadata', {}).get('agent_used', 'ResearcherAgent')}",
                        "response": result['response'],
                        "processing_time": result.get('processing_time', 0),
                        "user_type": "researcher"
                    }
                    
                    # Include visualization if available
                    if result.get('has_visualization') and result.get('visualization_html'):
                        response_data['visualization_html'] = result['visualization_html']
                        response_data['has_visualization'] = True
                    
                    return jsonify(response_data)
                else:
                    logger.error(f"❌ Researcher query error for {user_id}: {result.get('error', 'Unknown')}")
                    return jsonify(result)
            
            else:
                # Handle patient queries (default)
                result = await self.patient_agent.process_query(message, user_id)
                
                if result['success']:
                    logger.info(f"✅ Patient chat response sent to {user_id} via {result['agent_used']}")
                    if result.get('paraphrased_query'):
                        logger.info(f"🔄 Query paraphrased: '{message}' → '{result['paraphrased_query']}'")
                    
                    # Add user_type to response
                    result['user_type'] = 'patient'
                else:
                    logger.error(f"❌ Patient chat error for {user_id}: {result.get('error', 'Unknown error')}")
                
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
        try:
            with open('templates/login.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Return a simple login page if template not found
            return "File Not Found: templates/login.html. Please ensure the file exists."
    
    def get_chat_html(self):
        """Return the chat HTML page"""
        try:
            with open('templates/chat.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Return a simple chat page if template not found
            return "File Not Found: templates/chat.html. Please ensure the file exists."
    
    def get_researcher_chat_html(self):
        """Return the researcher chat HTML page"""
        try:
            with open('templates/researcher_chat.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Return a simple error page if template not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - Power of Patients</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 50px; text-align: center; }
                    h1 { color: #e74c3c; }
                    a { color: #3498db; text-decoration: none; }
                </style>
            </head>
            <body>
                <h1>File Not Found</h1>
                <p>researcher_chat.html not found in templates directory.</p>
                <p>Please ensure the file exists at: templates/researcher_chat.html</p>
                <a href="/">Return to Login</a>
            </body>
            </html>
            """
    # Add this to your main.py setup_routes method in the PowerOfPatientsApp class:

        # Add logout API route
        @app.route('/api/logout', methods=['POST'])
        def api_logout():
            return asyncio.run(self.handle_logout())

        # And add this method to your PowerOfPatientsApp class:

        async def handle_logout(self):
            """Handle user logout and cleanup connections"""
            try:
                # Get session ID from request if available
                data = request.get_json() or {}
                user_id = data.get('patient_id') or data.get('user_id')
                user_type = data.get('user_type', 'unknown')
                
                logger.info(f"Logout request from {user_type} {user_id}")
                
                # Remove from active sessions if being tracked
                for session_id, session_data in list(active_sessions.items()):
                    if session_data.get('user_id') == user_id:
                        del active_sessions[session_id]
                        logger.info(f"Removed session for {user_id}")
                        break
                try:
                    if user_type == 'researcher':
                        # Clear researcher agent memory if it has the method
                        if hasattr(self.researcher_agent, 'clear_conversation_history'):
                            await self.researcher_agent.clear_conversation_history(user_id)
                        else:
                            logger.info("Researcher agent doesn't have clear_conversation_history method")
                    else:
                        # Clear patient agent memory
                        await self.patient_agent.clear_conversation_history(user_id)
                        
                    logger.info(f"✅ Cleared conversation memory for {user_type} {user_id}")
                    
                except Exception as memory_error:
                    logger.error(f"Error clearing memory for {user_id}: {memory_error}")
                    # Don't fail the logout if memory clearing fails
                
                return jsonify({
                    "success": True,
                    "message": "Logged out successfully"
                })
                # Don't close connections here - we'll manage them with ensure_connection
                # when the user tries to log in again
                    
            except Exception as e:
                logger.exception(f"Error in logout: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "message": "Error during logout"
                }), 500
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
    print("🏥 POWER OF PATIENTS - INTEGRATED HEALTHCARE PLATFORM")
    print("=" * 60)
    print("Supporting both Patients and Researchers")
    print("=" * 60)
    print("Starting application server...")
    
    # Initialize connections
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    startup_success = loop.run_until_complete(startup())
    
    if startup_success:
        print("\n✅ Server initialized successfully!")
        print("\n📋 Available endpoints:")
        print("   • http://localhost:5000/                    - Login page")
        print("   • http://localhost:5000/chat.html           - Patient chat interface")
        print("   • http://localhost:5000/researcher_chat.html - Researcher chat interface")
        print("   • http://localhost:5000/api/health          - Health check")
        print("\n🔐 Authentication API:")
        print("   • POST /api/authenticate - Login (checks patient first, then researcher)")
        print("   • POST /api/chat        - Send message to appropriate agent")
        print("   • GET  /api/researcher/examples - Get research query examples")
        print("\n💡 Usage:")
        print("   1. Go to http://localhost:5000/")
        print("   2. Login with patient OR researcher credentials")
        print("   3. System automatically redirects to appropriate interface")
        print("   4. Patients → /chat.html")
        print("   5. Researchers → /researcher_chat.html")
        print("\n🤖 Agents:")
        print("   • Patients → Sallie (Medical guidance, TBI info)")
        print("   • Researchers → Sallie (Data analysis, visualizations)")
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
        print("   • Ensure patient_auth table exists with columns: email, password, patient_id, user_type")
        print("   • Check that patient_summary table exists for patient agent")
        print("   • Ensure templates/researcher_chat.html exists")

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
                
                # Test patient auth first
                print("\n1️⃣ Testing patient authentication...")
                patient_result = await patient_authenticator.authenticate_patient(test_email, test_password)
                
                if patient_result['success']:
                    print(f"✅ Patient authentication successful!")
                    print(f"   Patient ID: {patient_result['patient_id']}")
                    print(f"   Email: {patient_result['email']}")
                else:
                    print(f"❌ Patient authentication failed: {patient_result['message']}")
                    
                    # Test researcher auth
                    print("\n2️⃣ Testing researcher authentication...")
                    researcher_result = await researcher_authenticator.authenticate_researcher(test_email, test_password)
                    
                    if researcher_result['success']:
                        print(f"✅ Researcher authentication successful!")
                        print(f"   Researcher ID: {researcher_result['researcher_id']}")
                        print(f"   Email: {researcher_result['email']}")
                        print(f"   User Type: {researcher_result['user_type']}")
                    else:
                        print(f"❌ Researcher authentication failed: {researcher_result['message']}")
            
            asyncio.run(test())
            
        elif command == "test-chat":
            # Test chat system
            async def test_chat():
                await startup()
                
                user_id = input("Enter user ID (patient_id or researcher email): ").strip()
                user_type = input("Enter user type (patient/researcher): ").strip().lower()
                
                while True:
                    message = input(f"\n[{user_type} {user_id}] Your message (or 'quit'): ").strip()
                    if message.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if user_type == 'researcher':
                        result = await researcher_agent.process_query(message, user_id)
                    else:
                        result = await patient_agent.process_query(message, user_id)
                    
                    if result['success']:
                        print(f"\n🤖 Sallie: {result['response']}")
                        if result.get('has_visualization'):
                            print("📊 [Visualization would appear here in web interface]")
                        if result.get('paraphrased_query'):
                            print(f"🔄 Enhanced: '{message}' → '{result['paraphrased_query']}'")
                    else:
                        print(f"❌ Error: {result.get('error')}")
            
            asyncio.run(test_chat())
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  • python main.py              - Start web server")
            print("  • python main.py test-auth    - Test authentication")
            print("  • python main.py test-chat    - Test chat system")
    else:
        # Start web server
        run_development_server()