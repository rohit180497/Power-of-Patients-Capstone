import logging
import re
import json
from typing import Tuple, Optional
import asyncio
import google.generativeai as genai
from core.config import GEMINI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

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

async def parse_patient_input(input_text: str) -> Tuple[str, str, str]:
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

def basic_parse_input(input_text: str) -> Tuple[str, str, str]:
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

def generate_bot_response(message: str, context: dict, history: list) -> str:
    """
    Generate a bot response based on the patient's message and context
    
    Args:
        message: Patient's message
        context: Patient context data
        history: Chat history
        
    Returns:
        str: Bot response
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