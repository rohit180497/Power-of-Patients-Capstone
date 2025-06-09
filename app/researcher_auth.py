"""
Simplified Researcher Authentication System for Power of Patients
Handles researcher login verification only
"""

import os
import asyncio
import psycopg2
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from datetime import datetime

logger = logging.getLogger(__name__)

class ResearcherAuthenticator:
    """
    Simplified Researcher Authentication System
    Only handles login verification
    """
    
    def __init__(self):
        """Initialize the authentication system"""
        load_dotenv()
        
        # Database connection
        self.db_connection = None
        self.db_config = None  # Store config for reconnection
        
        logger.info("Researcher Authenticator initialized")
    
    async def connect_to_database(self, db_config: Dict[str, str] = None) -> bool:
        """Connect to database asynchronously"""
        try:
            if db_config is None:
                db_config = {
                    'user': os.getenv("user") or os.getenv("DB_USER"),
                    'password': os.getenv("password") or os.getenv("DB_PASSWORD"),
                    'host': os.getenv("host") or os.getenv("DB_HOST"),
                    'port': os.getenv("port") or os.getenv("DB_PORT", "5432"),
                    'dbname': os.getenv("dbname") or os.getenv("DB_NAME")
                }
            
            # Store config for reconnection
            self.db_config = db_config
            
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration keys: {missing_keys}")
                return False
            
            # Close existing connection if any
            if self.db_connection:
                try:
                    self.db_connection.close()
                except:
                    pass  # Ignore errors on closing
                    
            # Use asyncio to run database connection in thread pool
            loop = asyncio.get_event_loop()
            self.db_connection = await loop.run_in_executor(
                None, lambda: psycopg2.connect(**db_config)
            )
            
            logger.info("Successfully connected to the authentication database")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    async def ensure_connection(self) -> bool:
        """Ensure database connection is active, reconnect if needed"""
        try:
            # Check if connection is closed or in an error state
            if not self.db_connection or self.db_connection.closed:
                logger.info("Database connection is closed, reconnecting...")
                return await self.connect_to_database(self.db_config)
            
            # Test if connection is still working with a simple query
            cursor = self.db_connection.cursor()
            try:
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            except (psycopg2.Error, Exception) as e:
                logger.warning(f"Connection test failed: {e}")
                cursor.close()
                # Connection is not working, reconnect
                try:
                    self.db_connection.close()
                except:
                    pass  # Ignore errors on closing
                self.db_connection = None
                return await self.connect_to_database(self.db_config)
                
        except Exception as e:
            logger.error(f"Error ensuring database connection: {e}")
            self.db_connection = None
            return await self.connect_to_database(self.db_config)
    
    async def authenticate_researcher(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate researcher credentials with proper connection management
        
        Args:
            email: Researcher's email address
            password: Researcher's password (should be numeric)
            
        Returns:
            Dict with authentication result
        """
        start_time = datetime.now()
        fresh_connection = False
        
        try:
            # Create a fresh connection for this authentication attempt
            if not await self.connect_to_database():
                return {
                    "success": False,
                    "error": "Database connection not available",
                    "message": "Authentication service unavailable. Please try again."
                }
            fresh_connection = True
            
            # Validate input
            if not email or not password:
                return {
                    "success": False,
                    "error": "Missing credentials",
                    "message": "Please provide both email and password."
                }
            
            # Clean email input
            email = email.strip().lower()
            
            # Modified query to get the password and compare it in Python
            # This avoids type conversion issues in SQL
            auth_query = """
                SELECT email, password, user_type
                FROM patient_auth 
                WHERE email = %s AND user_type = 'researcher'
            """
            
            # Execute authentication query with proper transaction handling
            loop = asyncio.get_event_loop()
            
            def execute_auth_query():
                result = None
                cursor = None
                try:
                    cursor = self.db_connection.cursor()
                    cursor.execute(auth_query, (email,))
                    result = cursor.fetchone()
                    
                    if not result:
                        # Explicitly rollback on auth failure (no matching user)
                        self.db_connection.rollback()
                        return None
                    
                    email_db, password_db, user_type = result
                    
                    # Compare passwords as integers
                    try:
                        # Convert input password to int for comparison
                        numeric_password = int(password)
                        
                        # If passwords match
                        if numeric_password == password_db:
                            # Explicitly commit the transaction
                            self.db_connection.commit()
                            return (email_db, password_db, user_type)
                        else:
                            # Explicitly rollback on auth failure (wrong password)
                            self.db_connection.rollback()
                            return None
                    except ValueError:
                        # If password can't be converted to int, it can't match
                        # Explicitly rollback on auth failure
                        self.db_connection.rollback()
                        return None
                        
                except Exception as e:
                    # Rollback on error
                    if self.db_connection:
                        try:
                            self.db_connection.rollback()
                        except:
                            pass
                    raise e
                finally:
                    # Always close the cursor
                    if cursor:
                        cursor.close()
            
            auth_result = await loop.run_in_executor(None, execute_auth_query)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if auth_result:
                # Authentication successful
                email_db, password_db, user_type = auth_result
                
                logger.info(f"Researcher authenticated successfully: {email}")
                
                return {
                    "success": True,
                    "researcher_id": email,  # Using email as researcher_id
                    "email": email_db,
                    "user_type": user_type,
                    "message": "Authentication successful",
                    "processing_time": processing_time
                }
            else:
                # Authentication failed
                logger.warning(f"Authentication failed for email: {email}")
                
                return {
                    "success": False,
                    "error": "Invalid credentials",
                    "message": "Invalid email or password. Please check your credentials and try again.",
                    "processing_time": processing_time
                }
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Error during authentication: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred during authentication. Please try again.",
                "processing_time": processing_time
            }
        finally:
            # Always close the connection when done if we created a fresh one
            if fresh_connection and self.db_connection:
                try:
                    self.db_connection.close()
                    self.db_connection = None
                    logger.debug("Closed database connection after authentication")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
    
    async def close_connection(self):
        """Safely close the database connection"""
        if self.db_connection:
            try:
                self.db_connection.close()
                self.db_connection = None
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")


# Test the authentication system
async def test_authentication():
    """Test function to verify researcher authentication system"""
    print("=" * 50)
    print("TESTING RESEARCHER AUTHENTICATION SYSTEM")
    print("=" * 50)
    
    # Initialize authenticator
    auth = ResearcherAuthenticator()
    
    # Connect to database
    print("Connecting to database...")
    if not await auth.connect_to_database():
        print("❌ Failed to connect to database")
        return
    
    print("✅ Connected to database successfully")
    
    # Test authentication
    test_email = input("Enter test researcher email: ").strip()
    test_password = input("Enter test password: ").strip()
    
    print(f"\nTesting authentication for: {test_email}")
    result = await auth.authenticate_researcher(test_email, test_password)
    
    if result["success"]:
        print("✅ Authentication successful!")
        print(f"Researcher ID: {result['researcher_id']}")
        print(f"Email: {result['email']}")
        print(f"User Type: {result['user_type']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
    else:
        print("❌ Authentication failed!")
        print(f"Error: {result['error']}")
        print(f"Message: {result['message']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
    
    # Test connection management
    print("\nTesting connection management...")
    print("Simulating connection issue by closing connection...")
    await auth.close_connection()
    
    print("Attempting authentication after connection closed...")
    result = await auth.authenticate_researcher(test_email, test_password)
    
    if result["success"]:
        print("✅ Connection recovery successful!")
        print(f"Authentication successful even after connection was closed")
    else:
        print("❌ Connection recovery failed!")
        print(f"Error: {result['error']}")
        print(f"Message: {result['message']}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_authentication())