import os
import datetime
import re
import logging
from typing import Tuple, Optional, Dict, Any, List
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PatientVerifier:
    """
    A utility class for verifying patient identity in the database
    using LLM for processing patient information.
    """
    
    def __init__(self, db_config: Dict[str, str] = None, gemini_api_key: str = None):
        """
        Initialize the PatientVerifier
        
        Args:
            db_config (Dict[str, str], optional): Database connection configuration
            gemini_api_key (str, optional): API key for Google's Gemini model
        """
        # Load config from environment if not provided
        if db_config is None:
            self.db_config = {
                'user': os.getenv("user"),
                'password': os.getenv("password"),
                'host': os.getenv("host"),
                'port': os.getenv("port"),
                'dbname': os.getenv("dbname")
            }
        else:
            self.db_config = db_config
            
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.connection = None
        
        # Configure Gemini if API key is provided
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            
            # Configure safety settings
            self.safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
    
    def connect_to_database(self) -> bool:
        """
        Connect to the PostgreSQL database
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Check if already connected
            if self.connection and not self.connection.closed:
                return True
                
            # Connect to the database
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = True
            
            logger.info("Successfully connected to the database")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    def close_connection(self):
        """Close the database connection if open"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    async def parse_patient_input(self, input_text: str) -> Tuple[str, str, str]:
        """
        Use LLM to parse potentially ambiguous patient input
        
        Args:
            input_text (str): Raw input text containing name and DOB information
            
        Returns:
            Tuple[str, str, str]: Parsed (first_name, last_name, dob)
        """
        if not self.gemini_api_key:
            # Fallback to basic parsing if no Gemini API key
            return self._basic_parse_input(input_text)
        
        try:
            # Initialize the model with high determinism
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "top_k": 1,
                    "max_output_tokens": 100,
                },
                safety_settings=self.safety_settings
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
            return self._basic_parse_input(input_text)
            
        except Exception as e:
            logger.exception(f"Error with Gemini parsing: {str(e)}")
            return self._basic_parse_input(input_text)
    
    def _basic_parse_input(self, input_text: str) -> Tuple[str, str, str]:
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
    
    async def verify_patient_in_database(self, input_text: str) -> Optional[str]:
        """
        Verify if a patient exists in the database using the input text
        
        Args:
            input_text (str): Raw input text containing patient identifiers
            
        Returns:
            Optional[str]: Patient ID if found, None otherwise
        """
        if not self.connect_to_database():
            logger.error("Cannot connect to database")
            return None
        
        try:
            # Parse the input text to extract first name, last name, and DOB
            first_name, last_name, dob = await self.parse_patient_input(input_text)
            
            if not first_name or not last_name or not dob:
                logger.warning(f"Incomplete patient information parsed: {first_name}, {last_name}, {dob}")
                return None
            
            # Query the database
            cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
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
            
            if result:
                return result['patient_id']
            
            logger.warning(f"No patient found with parsed data: {first_name}, {last_name}, {dob}")
            return None
            
        except Exception as e:
            logger.exception(f"Error verifying patient: {str(e)}")
            return None
    
    async def verify_patient(self, first_name: str, last_name: str, dob: str) -> Optional[str]:
        """
        Verify if a patient exists in the database using structured inputs
        
        Args:
            first_name (str): Patient's first name
            last_name (str): Patient's last name
            dob (str): Patient's date of birth in any format
            
        Returns:
            Optional[str]: Patient ID if found, None otherwise
        """
        # Combine inputs into a single text string for processing
        input_text = f"{first_name} {last_name}, {dob}"
        return await self.verify_patient_in_database(input_text)

# For direct testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test the patient verifier
        verifier = PatientVerifier()
        
        # Example 1: Normal input
        patient_id = await verifier.verify_patient("caroline", "suong", "1981-12-09")
        print(f"Example 1 result: {patient_id}")
        
        # Example 2: Ambiguous input
        patient_id = await verifier.verify_patient_in_database("solo, mike - 12th June 1987")
        print(f"Example 2 result: {patient_id}")
        
        # Example 3: Name variant
        patient_id = await verifier.verify_patient_in_database("I am neil langrick born on november 30 , 1963")
        print(f"Example 3 result: {patient_id}")
        
        # Close the database connection
        verifier.close_connection()

    # Run the test function
    asyncio.run(main())