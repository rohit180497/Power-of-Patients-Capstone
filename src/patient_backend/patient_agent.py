import os
import json
import logging
import asyncio
import psycopg2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Optional, Any, Tuple, List
from dotenv import load_dotenv
from datetime import datetime
import re
from collections import deque

# Import your existing agents - commented out for now
# Uncomment these when you're ready to use the actual agents
from src.medpalm.medical_assistant import MedPalmAgent
from src.retrieval.cdcretrieval import CDCTBIRetriever
from src.guard.guardrailagent import MedicalGuardrailAgent

logger = logging.getLogger(__name__)

class ProfessionalPatientAgent:
    """
    Professional Patient Agent that acts as a router and coordinator
    for patient healthcare conversations
    """
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None, 
                 index_name: str = "us-cdc-tbi", embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_provider: str = "gemini"):
        """Initialize the Professional Patient Agent with all sub-agents"""
        
        # Load environment variables
        load_dotenv()
        
        # Store configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX2_NAME")
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.llm_provider = "gemini"
        
        # Initialize database connection
        self.db_connection = None
        
        # Patient data storage
        self.current_patient_data = {}
        self.patient_context = ""
        self.session_history = deque(maxlen=10)  # Memory buffer using deque
        self.welcomed_patients = set()  # Track patients who have been welcomed
        
        # Conversational Memory Components
        self.pending_questions = {}  # Track questions waiting for responses
        self.last_assistant_message = ""  # Track what assistant said last

        # Initialize guardrail agent
        self.guardrail_agent = MedicalGuardrailAgent(self.gemini_api_key)
        
        # Pre-initialize all agents to avoid delays
        self._initialize_agents()
        
        logger.info("Professional Patient Agent initialized successfully")
    
    def _initialize_agents(self):
        """Pre-initialize all sub-agents for fast response times"""
        try:
            # Initialize Gemini for routing and context management
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini 2.0 Flash model initialized")
            
            # Initialize Medical Guardrail Agent
            try:
                self.guardrail_agent = MedicalGuardrailAgent(self.gemini_api_key)
                logger.info("Medical Guardrail Agent initialized")
            except Exception as e:
                logger.warning(f"Guardrail Agent not available: {e}")
                self.guardrail_agent = None
            
            # Initialize MedPalm Agent
            try:
                
                self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
                
                logger.info("MedPalm Agent placeholder initialized (update imports to use real agent)")
                
            except Exception as e:
                logger.warning(f"MedPalm Agent not available: {e}")
                self.medical_agent = None
            
            # Initialize TBI Retrieval Agent
            try:
                
                self.retrieval_agent = CDCTBIRetriever(
                    pinecone_api_key=self.pinecone_api_key,
                    index_name=self.index_name,
                    embedding_model=self.embedding_model,
                    llm_provider=self.llm_provider
                )
                
                logger.info("TBI Retrieval Agent placeholder initialized (update imports to use real agent)")
                
            except Exception as e:
                logger.warning(f"TBI Retrieval Agent not available: {e}")
                self.retrieval_agent = None
                
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
    
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
            
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration keys: {missing_keys}")
                return False
            
            # Use asyncio to run database connection in thread pool
            loop = asyncio.get_event_loop()
            self.db_connection = await loop.run_in_executor(
                None, lambda: psycopg2.connect(**db_config)
            )
            
            logger.info("Successfully connected to the database")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    async def load_patient_data(self, patient_id: str) -> bool:
        """Load patient data using the single comprehensive query"""
        if not self.db_connection:
            logger.error("Database connection not established")
            return False
        
        try:
            query = """
            SELECT patient_id,
                   first_name, 
                   user_type, 
                   registered_at, 
                   country, 
                   referral_group, 
                   veteran, 
                   city,
                   patient_type,
                   patient_sub_type,
                   head_hit_location,
                   has_tbi_before,
                   age,
                   tbi_incident_date,
                   injury_from, 
                   head_hit_location,
                   num_head_hit_location,
                   total_tbi,
                   immediate_symptoms_resulting,
                   describe_event,
                   worst_symptoms,
                   symptom_json,
                   sdoh_json,
                   therapy_json
            FROM patient_summary
            WHERE patient_id = %s
            """
            
            # Execute query asynchronously
            loop = asyncio.get_event_loop()
            
            def execute_query():
                cursor = self.db_connection.cursor()
                cursor.execute(query, (patient_id,))
                result = cursor.fetchone()
                
                if result:
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    patient_data = dict(zip(columns, result))
                    cursor.close()
                    return patient_data
                else:
                    cursor.close()
                    return None
            
            patient_data = await loop.run_in_executor(None, execute_query)
            
            if not patient_data:
                logger.error(f"No patient found with ID: {patient_id}")
                return False
            
            self.current_patient_data = patient_data
            await self._build_patient_context()
            
            logger.info(f"Successfully loaded data for patient ID: {patient_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading patient data: {str(e)}")
            return False
    
    async def _build_patient_context(self):
        """Build comprehensive patient context for LLM memory"""
        if not self.current_patient_data:
            return
        
        data = self.current_patient_data
        context_parts = []
        
        # Basic Patient Information
        context_parts.append("=== PATIENT PROFILE ===")
        context_parts.append(f"Patient ID: {data.get('patient_id', 'N/A')}")
        context_parts.append(f"Name: {data.get('first_name', 'N/A')}")
        context_parts.append(f"Age: {data.get('age', 'N/A')}")
        context_parts.append(f"User Type: {data.get('user_type', 'N/A')}")
        context_parts.append(f"Patient Type: {data.get('patient_type', 'N/A')}")
        context_parts.append(f"Patient Sub-Type: {data.get('patient_sub_type', 'N/A')}")
        context_parts.append(f"Location: {data.get('city', 'N/A')}, {data.get('country', 'N/A')}")
        context_parts.append(f"Veteran Status: {data.get('veteran', 'N/A')}")
        context_parts.append(f"Referral Group: {data.get('referral_group', 'N/A')}")
        context_parts.append(f"Registered: {data.get('registered_at', 'N/A')}")
        
        # TBI History and Incident Details
        context_parts.append("\n=== TBI INCIDENT INFORMATION ===")
        context_parts.append(f"Previous TBI History: {data.get('has_tbi_before', 'N/A')}")
        context_parts.append(f"Total TBI Count: {data.get('total_tbi', 'N/A')}")
        context_parts.append(f"Incident Date: {data.get('tbi_incident_date', 'N/A')}")
        context_parts.append(f"Injury Source: {data.get('injury_from', 'N/A')}")
        context_parts.append(f"Head Hit Location: {data.get('head_hit_location', 'N/A')}")
        context_parts.append(f"Number of Head Hit Locations: {data.get('num_head_hit_location', 'N/A')}")
        context_parts.append(f"Event Description: {data.get('describe_event', 'N/A')}")
        
        # Symptoms
        context_parts.append("\n=== SYMPTOM INFORMATION ===")
        context_parts.append(f"Immediate Symptoms: {data.get('immediate_symptoms_resulting', 'N/A')}")
        context_parts.append(f"Worst Symptoms: {data.get('worst_symptoms', 'N/A')}")
        
        # Parse JSON fields if available
        if data.get('symptom_json'):
            try:
                symptom_data = json.loads(data['symptom_json']) if isinstance(data['symptom_json'], str) else data['symptom_json']
                context_parts.append(f"Detailed Symptoms: {json.dumps(symptom_data, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                context_parts.append(f"Symptom Details: {data['symptom_json']}")
        
        # Social Determinants of Health
        if data.get('sdoh_json'):
            try:
                sdoh_data = json.loads(data['sdoh_json']) if isinstance(data['sdoh_json'], str) else data['sdoh_json']
                context_parts.append("\n=== SOCIAL DETERMINANTS OF HEALTH ===")
                context_parts.append(f"SDOH Details: {json.dumps(sdoh_data, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                context_parts.append(f"SDOH Details: {data['sdoh_json']}")
        
        # Therapy Information
        if data.get('therapy_json'):
            try:
                therapy_data = json.loads(data['therapy_json']) if isinstance(data['therapy_json'], str) else data['therapy_json']
                context_parts.append("\n=== THERAPY AND TREATMENT ===")
                context_parts.append(f"Therapy Details: {json.dumps(therapy_data, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                context_parts.append(f"Therapy Details: {data['therapy_json']}")
        
        self.patient_context = "\n".join(context_parts)
    
    def _get_time_based_greeting(self) -> str:
        """Generate time-based greeting based on current time"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good Morning"
        elif 12 <= current_hour < 17:
            return "Good Afternoon"
        elif 17 <= current_hour < 22:
            return "Good Evening"
        else:
            return "Hello"
    
    def _generate_welcome_message(self, patient_name: str) -> str:
        """Generate personalized welcome message for new patients"""
        time_greeting = self._get_time_based_greeting()
        
        welcome_message = f"""Hello {patient_name}, {time_greeting}! 

I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. I'm here to help you with your medical questions, TBI information, and healthcare guidance.

How can I assist you with your health concerns today?"""
        
        return welcome_message
    
    def _format_recent_history(self, num_entries=3):
        """Format recent conversation history for context"""
        history_parts = []
        for exchange in list(self.session_history)[-num_entries:]:
            history_parts.append(f"Patient: {exchange['query']}")
            history_parts.append(f"Assistant: {exchange['response'][:100]}...")
        return "\n".join(history_parts)
    
    def _update_conversation_memory(self, user_query: str, assistant_response: str):
        """Update conversational memory with new exchange"""
        # Add to session history (deque automatically manages size)
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": self.current_patient_data.get('patient_id'),
            "query": user_query,
            "response": assistant_response,
            "agent_used": getattr(self, '_last_agent_used', 'Unknown')
        }
        self.session_history.append(exchange)
        
        # Update last assistant message
        self.last_assistant_message = assistant_response
        
        # Check if assistant asked a question and track it
        self._track_pending_questions(assistant_response)
    
    def _track_pending_questions(self, assistant_response: str):
        """Track questions that the assistant asked and are awaiting responses"""
        # Simple question detection patterns
        question_patterns = [
            r"do you want me to.*\?",
            r"would you like.*\?", 
            r"should i.*\?",
            r"are you.*\?",
            r"can i.*\?",
            r"shall i.*\?",
            r"need.*help.*\?",
            r"want.*contact.*\?"
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, assistant_response.lower()):
                # Store the question with timestamp
                self.pending_questions["last_question"] = {
                    "question": assistant_response,
                    "timestamp": datetime.now().isoformat(),
                    "type": "yes_no_question"
                }
                break
    
    def _is_continuation_response(self, query: str) -> bool:
        """Check if the query is a continuation response (like 'Yes', 'No', 'Sure', etc.)"""
        query_lower = query.lower().strip()
        
        # Common continuation responses
        continuation_patterns = [
            r"^(yes|yeah|yep|sure|okay|ok|fine|alright)\.?$",
            r"^(no|nope|nah|not really)\.?$", 
            r"^(maybe|perhaps|i think so)\.?$",
            r"^(please|go ahead|that would be great)\.?$",
            r"^(thanks|thank you|appreciate it)\.?$"
        ]
        
        for pattern in continuation_patterns:
            if re.match(pattern, query_lower):
                return True
        
        return False
    
    async def _paraphrase_query(self, query: str, intent: str = "general") -> str:
        """
        Paraphrase user query using chat history context to resolve ambiguity and pronouns
        """
        try:
            # Skip paraphrasing for certain types of queries
            skip_paraphrasing_intents = ["crisis", "continuation_response"]
            if intent in skip_paraphrasing_intents:
                logger.info(f"Skipping paraphrasing for intent: {intent}")
                return query
            
            # Skip paraphrasing for very short simple queries without pronouns
            if len(query.strip().split()) <= 2 and not any(pronoun in query.lower() for pronoun in ['my', 'mine', 'they', 'it', 'this', 'that', 'these', 'those']):
                logger.info("Skipping paraphrasing for short simple query")
                return query
            
            # Get recent conversation history for context
            recent_history = self._format_recent_history(num_entries=3) if self.session_history else ""
            
            # Get patient name for personalization
            patient_name = self.current_patient_data.get('first_name', 'the patient')
            
            # Build paraphrasing prompt
            paraphrasing_prompt = f"""
You are a query paraphrasing assistant for a healthcare chatbot. Your job is to rewrite user queries to be more explicit and clear while preserving the original intent.

PATIENT NAME: {patient_name}

{"RECENT CONVERSATION CONTEXT:" if recent_history else ""}
{recent_history}

ORIGINAL USER QUERY: "{query}"

PARAPHRASING RULES:
1. Resolve pronouns and ambiguous references using conversation context
2. Make implicit medical topics explicit based on conversation history
3. Replace "my", "mine" with "{patient_name}'s" when referring to medical information
4. Replace "they", "it", "this", "that" with specific subjects from context
5. Preserve the original medical intent and urgency
6. Keep the tone natural and conversational
7. If the query is already clear and specific, change it minimally
8. Use conversation context to infer what pronouns refer to

CONTEXT-BASED EXAMPLES:
- "how they are different from mine?" (after TBI discussion) → "how are other people's TBI symptoms different from {patient_name}'s TBI symptoms?"
- "my symptoms" (in medical context) → "{patient_name}'s symptoms"
- "what about mine?" (after discussing conditions) → "what about {patient_name}'s condition?"
- "i am having headache" → "{patient_name} is having a headache"
- "tell me about myself" → "tell me about {patient_name}'s medical information and health profile"
- "they seem worse" (after symptom discussion) → "the symptoms seem worse"

IMPORTANT: 
- Only paraphrase if it adds clarity or resolves ambiguity
- Don't change medical terminology unnecessarily
- Preserve question marks and punctuation
- Keep the same level of formality
- Use conversation context to understand what pronouns mean

Paraphrased query:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    paraphrasing_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            paraphrased_query = response.text.strip()
            
            # Clean up the response (remove quotes if wrapped)
            if paraphrased_query.startswith('"') and paraphrased_query.endswith('"'):
                paraphrased_query = paraphrased_query[1:-1]
            
            # Validate paraphrased query
            if len(paraphrased_query) > 0 and len(paraphrased_query) < 500:  # Reasonable length
                logger.info(f"Query paraphrased: '{query}' → '{paraphrased_query}'")
                return paraphrased_query
            else:
                logger.warning(f"Paraphrasing failed or too long, using original query")
                return query
            
        except Exception as e:
            logger.error(f"Error in query paraphrasing: {e}")
            return query  # Return original query if paraphrasing fails
    
    async def _classify_query_intent(self, query: str) -> str:
        """Use LLM to intelligently classify query intent with conversational context"""
        try:
            # Check if this is a continuation response first
            if self._is_continuation_response(query) and self.pending_questions:
                # This is likely a response to a previous question
                logger.info(f"Detected continuation response: {query}")
                return "continuation_response"
            
            # Get recent conversation context
            recent_history = self._format_recent_history(num_entries=3) if self.session_history else ""
            
            # Build classification prompt with conversational context
            classification_prompt = f"""
You are a healthcare query classifier. Analyze the user's query considering the conversation context.

{"RECENT CONVERSATION:" if recent_history else ""}
{recent_history}

USER QUERY: "{query}"

ROUTING OPTIONS:
1. "crisis" - For mental health crises, suicide ideation, self-harm, or emergency situations
2. "medical_general" - For general medical questions (diabetes, heart disease, medications, treatments, etc.)
3. "tbi_retrieval" - For questions specifically about TBI, concussion, brain injury, head trauma, or brain injury symptoms  
4. "continuation_response" - For responses like "yes", "no", "sure" that continue previous conversation
5. "general_conversation" - For greetings, thanks, casual conversation, or non-medical questions

CLASSIFICATION RULES:
- CRISIS queries (suicide, self-harm, emergency) get highest priority
- If the query is a short response like "yes/no/sure" and follows a question, classify as "continuation_response"
- TBI queries are specifically about traumatic brain injury, concussion, or brain trauma
- Medical queries are about other health conditions, medications, treatments
- General conversation is for non-medical social interaction

IMPORTANT: Respond with ONLY one of these options: "crisis", "medical_general", "tbi_retrieval", "continuation_response", or "general_conversation"

Classification:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    classification_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            # Extract and validate the classification
            classification = response.text.strip().lower()
            
            # Validate response and provide fallback
            valid_classifications = ["crisis", "medical_general", "tbi_retrieval", "continuation_response", "general_conversation"]
            
            for valid_class in valid_classifications:
                if valid_class in classification:
                    logger.info(f"Query classified as: {valid_class}")
                    return valid_class
            
            # Fallback logic with enhanced context consideration
            query_lower = query.lower()
            
            # Crisis keywords get highest priority
            crisis_indicators = ['suicide', 'kill myself', 'self-harm', 'hurt myself', 'end my life', 'emergency', 'crisis']
            if any(indicator in query_lower for indicator in crisis_indicators):
                logger.warning("Crisis detected in fallback classification")
                return "crisis"
            
            # Check for continuation response in fallback
            if self._is_continuation_response(query) and self.pending_questions:
                return "continuation_response"
            
            # TBI-specific indicators
            tbi_indicators = ['tbi', 'concussion', 'brain injury', 'head trauma', 'brain trauma']
            if any(indicator in query_lower for indicator in tbi_indicators):
                return "tbi_retrieval"
            
            # General medical indicators
            medical_indicators = ['diabetes', 'heart', 'blood pressure', 'medication', 'prescription', 'doctor', 'treatment']
            if any(indicator in query_lower for indicator in medical_indicators):
                return "medical_general"
            
            return "general_conversation"
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            
            # Emergency fallback with conversation awareness
            query_lower = query.lower()
            
            # Crisis detection
            crisis_indicators = ['suicide', 'kill myself', 'self-harm', 'hurt myself', 'end my life']
            if any(indicator in query_lower for indicator in crisis_indicators):
                return "crisis"
            
            # Continuation response detection
            if self._is_continuation_response(query) and self.pending_questions:
                return "continuation_response"
            
            return "general_conversation"
    
    async def _handle_crisis(self, query: str) -> str:
        """Handle mental health crisis queries with immediate support"""
        crisis_response = """
I'm very concerned about what you're sharing with me. Your life has value and there are people who want to help you.

🚨 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• International: befrienders.org

Please reach out to:
• Emergency services (911) if in immediate danger
• Your healthcare provider
• A trusted friend or family member
• Local crisis center

You don't have to go through this alone. Professional counselors are available 24/7 to provide support and help you work through these feelings.

Is there someone you can call right now? Would you like help finding local mental health resources?
"""
        return crisis_response
    
    async def _handle_continuation_response(self, query: str) -> str:
        """Handle continuation responses like 'Yes', 'No' based on conversation context"""
        try:
            if not self.pending_questions:
                return "I'm not sure what you're referring to. Could you please clarify your question?"
            
            # Get the last question and context
            last_question_info = self.pending_questions.get("last_question", {})
            last_question = last_question_info.get("question", "")
            
            # Get recent conversation history
            recent_history = self._format_recent_history(num_entries=3)
            
            # Build context-aware response prompt
            continuation_prompt = f"""
You are a healthcare assistant continuing a conversation with a patient.

RECENT CONVERSATION:
{recent_history}

PREVIOUS ASSISTANT MESSAGE: {last_question}

PATIENT RESPONSE: "{query}"

The patient is responding to your previous message/question. Please provide an appropriate follow-up response that:
1. Acknowledges their response
2. Takes appropriate action based on their answer
3. Continues the conversation naturally
4. Provides helpful information or next steps

If they said "yes" to an offer of help, provide that help.
If they said "no", acknowledge and offer alternatives.
If unclear, ask for clarification.

Respond naturally as their healthcare assistant:
"""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.gemini_model.generate_content(
                    continuation_prompt, 
                    safety_settings=self.safety_settings
                )
            )
            
            # Clear the pending question since it's been addressed
            self.pending_questions.clear()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error handling continuation response: {e}")
            return "I understand you're responding to my previous message. Could you help me understand what specific information you need?"
    
    async def _call_medical_agent(self, query: str) -> str:
        """Call MedPalm agent for general medical queries"""
        try:
            if self.medical_agent:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.medical_agent.process_query(query)
                )
                return response
                
                # Mock response for now
                # return f"[MedPalm Agent Response] This is a medical response to: {query}"
            else:
                return "Medical agent not available. Please consult your healthcare provider."
                
        except Exception as e:
            logger.error(f"Error calling medical agent: {e}")
            return "I encountered an error accessing medical information. Please try again or consult your healthcare provider."
    
    async def _call_retrieval_agent(self, query: str) -> str:
        """Call TBI retrieval agent for TBI-related queries"""
        try:
            if self.retrieval_agent:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.retrieval_agent.ask_question(query, top_k=8)
                )
                return result['answer']
                
                # Mock response for now
                # return f"[TBI Retrieval Agent Response] This is a TBI-specific response to: {query}"
            else:
                return "TBI information system not available. Please consult your healthcare provider."
                
        except Exception as e:
            logger.error(f"Error calling retrieval agent: {e}")
            return "I encountered an error accessing TBI information. Please try again."
    
    def _is_medical_query(self, intent: str) -> bool:
        """Check if the query intent is medical-related"""
        return intent in ["medical_general", "tbi_retrieval"]
    
    def _get_medical_disclaimer(self, intent: str) -> str:
        """Get appropriate medical disclaimer based on intent"""
        if intent == "medical_general":
            return "\n\n⚠️ **Medical Disclaimer:** I am an AI assistant and cannot provide personalized medical advice, diagnoses, or treatment recommendations. This information is for educational purposes only and should not replace professional medical consultation. Please consult with your healthcare provider, doctor, or qualified medical professional for personalized medical advice, proper diagnosis, and treatment options specific to your condition."
        elif intent == "tbi_retrieval":
            return "\n\n⚠️ **Medical Disclaimer:** I am an AI assistant providing general information about traumatic brain injury. This information is educational and should not replace professional medical consultation. Please consult with your healthcare provider, neurologist, or qualified medical professional for personalized medical advice regarding brain injuries, proper diagnosis, and treatment options."
        else:
            return ""
    
    async def _generate_contextual_response(self, query: str, agent_response: str, intent: str) -> str:
        """Generate final response using patient context, chat history, and agent response with medical disclaimers"""
        try:
            # For crisis situations, return the crisis response directly without modification
            if intent == "crisis":
                return agent_response
            
            # For continuation responses, the agent already handled context, return as-is
            if intent == "continuation_response":
                return agent_response
            
            # Check if this is a medical query
            is_medical = self._is_medical_query(intent)
            
            # Get recent conversation history - but only if it's relevant to current query
            recent_history = self._format_recent_history(num_entries=2) if len(self.session_history) > 0 else ""
            
            # Determine if patient context is relevant based on intent and query
            context_relevant = self._is_patient_context_relevant(query, intent)
            
            # Check if this is truly the first interaction or a continuation
            is_first_interaction = len(self.session_history) == 0
            has_recent_history = bool(recent_history.strip())
            
            # # Check if this is a patient self-inquiry
            is_patient_self_inquiry = self._is_patient_self_inquiry(query)
            
            # Power of Patients company information
            power_of_patients_info = """
    ABOUT POWER OF PATIENTS:
    Power of Patients is a healthcare technology platform that empowers patients to take control of their health journey. The platform focuses on:
    - Patient empowerment and advocacy
    - Healthcare navigation and support
    - Medical information and resources
    - Connecting patients with healthcare providers
    - TBI and neurological condition support
    - Personalized healthcare guidance

    You (Sallie) are a healthcare assistant created by Power of Patients to help patients with medical questions, TBI information, and healthcare navigation.
    """
            
            # Only include Power of Patients info if the query is about the company
            include_company_info = any(term in query.lower() for term in ['power of patients', 'platform', 'company', 'sallie', 'who created', 'who made'])
            
            # Smart memory management: summarize older conversations when buffer is full
            if len(self.session_history) >= 10:
                summarized_memory = "\n".join(
                    f"Patient asked: {entry['query']}\nAssistant replied: {entry['response'][:80]}..."
                    for entry in list(self.session_history)[:-3]
                )
                memory_block = f"\nPAST CONVERSATION (SUMMARY):\n{summarized_memory}\n"
            else:
                memory_block = ""
            
            # Build smart system prompt with conversation awareness, memory management, and medical safety
            medical_safety_instructions = ""
            if is_medical:
                medical_safety_instructions = """
    CRITICAL MEDICAL SAFETY INSTRUCTIONS:
    - NEVER suggest specific medications, drugs, or prescriptions
    - NEVER provide specific dosages or treatment protocols
    - NEVER diagnose medical conditions
    - ALWAYS remind the patient to consult their healthcare provider
    - Provide general educational information only
    - Focus on encouraging professional medical consultation
    - If asked about medications, redirect to "consult your doctor or pharmacist"
    - For treatment questions, say "your healthcare provider can best advise you"
    """
            
            # Dynamic conversation context instruction - FIXED to prevent context bleeding
            conversation_context_instruction = ""
            if is_first_interaction:
                conversation_context_instruction = """
    IMPORTANT: This is a FRESH query. Do NOT reference any previous conversations or use phrases like "as we discussed," "as mentioned before," or carry over context from other queries. Respond to THIS specific query only.
    """
            elif has_recent_history:
                conversation_context_instruction = """
    CONVERSATION CONTEXT: You may reference the RECENT conversation history below ONLY if it's directly relevant to the current query. Do not mix contexts from unrelated previous queries.
    """
            else:
                conversation_context_instruction = """
    IMPORTANT: Treat this as a fresh query. Do not reference non-existent previous conversations.
    """
            
            # Patient information sharing instructions
            patient_info_instructions = ""
            if is_patient_self_inquiry:
                patient_info_instructions = """
    PATIENT SELF-INQUIRY DETECTED: The patient is asking about themselves or their medical information. You should:
    1. Share relevant information from their patient context
    2. Be empathetic and supportive
    3. Explain information in an understandable way
    4. Remind them this is their information from their profile
    5. Encourage them to discuss any concerns with their healthcare provider
    6. Respect their right to know their own medical information
    """
            elif context_relevant:
                patient_info_instructions = """
    PATIENT CONTEXT USAGE: Only reference patient information that is directly relevant to their current question.
    """
            
            system_prompt = f"""
    {memory_block}
    You are Sallie, a professional, empathetic healthcare assistant created by Power of Patients.

    {conversation_context_instruction}

    {power_of_patients_info if include_company_info else ""}

    {medical_safety_instructions}

    {patient_info_instructions}

    CURRENT USER QUERY: {query}
    SPECIALIST RESPONSE: {agent_response}

    {"RECENT RELEVANT CONVERSATION (only reference if directly related to current query):" if has_recent_history else ""}
    {recent_history if has_recent_history else ""}

    {"PATIENT CONTEXT - This is {self.current_patient_data.get('first_name', 'the patient')}'s medical information:" if context_relevant else ""}
    {self.patient_context if context_relevant else ""}

    CORE INSTRUCTIONS FOR THIS SPECIFIC QUERY:
    1. Answer the CURRENT query "{query}" specifically - do not mix in unrelated previous topics
    2. Be empathetic and professional
    3. {"If the patient is asking about themselves, share relevant information from their patient context in a caring, understandable way" if is_patient_self_inquiry else "Only reference previous conversations if they are directly related to the current query topic" if has_recent_history else "Do not reference any previous conversations"}
    4. {"Share the patient's medical information since they are asking about themselves - they have the right to know their own information" if is_patient_self_inquiry else "Reference the patient's medical history ONLY if directly relevant to their current question" if context_relevant else "Keep the response focused on their current question"}
    5. Do not overwhelm with unnecessary medical details
    6. For medical concerns, ALWAYS remind them to consult their healthcare provider
    7. Be concise and helpful
    8. If asking follow-up questions, make them clear and specific
    {"9. NEVER suggest medications or specific treatments - always direct to healthcare providers" if is_medical else ""}
    {"10. Focus on providing educational information while emphasizing professional medical consultation" if is_medical else ""}

    CRITICAL CONTEXT RULES:
    - Do NOT start responses with context from completely different queries
    - Do NOT say "as we discussed" unless the current query is directly continuing a previous topic
    - Each query should be treated as its own focused interaction
    - Only bridge to previous context if the topics are genuinely related

    {"COMPANY KNOWLEDGE: If asked about Power of Patients, registration dates, or platform information, use your knowledge about the company and platform." if include_company_info else ""}

    {"PATIENT PRIVACY: When sharing patient information, be respectful and remind them this information comes from their medical profile. Encourage them to discuss any questions with their healthcare provider." if is_patient_self_inquiry else ""}

    Respond naturally and helpfully to the current query:
    """
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.gemini_model.generate_content(
                    system_prompt, 
                    safety_settings=self.safety_settings
                )
            )
            
            # Get the generated response
            final_response = response.text.strip()
            
            # Add medical disclaimer for medical queries
            if is_medical:
                medical_disclaimer = self._get_medical_disclaimer(intent)
                final_response += medical_disclaimer
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            # Fallback with disclaimer if medical
            fallback_response = agent_response
            if self._is_medical_query(intent):
                fallback_response += self._get_medical_disclaimer(intent)
            return fallback_response
        

    def _is_patient_self_inquiry(self, query: str) -> bool:
        """Check if the patient is asking about themselves or their own medical information"""
        query_lower = query.lower().strip()
        
        # Self-inquiry patterns
        self_inquiry_patterns = [
            # Direct questions about self
            r'\b(tell me about myself|what do you know about me|my information|my profile)\b',
            r'\b(what do you know about (me|my case|my history))\b',
            r'\b(my medical history|my health record|my diagnosis|my condition)\b',
            r'\b(my symptoms|my injury|my accident|my tbi|my treatment)\b',
            r'\b(when did i register|my registration|registered date)\b',
            r'\b(about my case|my recovery|my progress|how am i doing)\b',
            r'\b(what.*you.*know.*about.*me|information.*about.*me)\b',
            r'\b(my.*data|my.*details|my.*file)\b',
            # Questions starting with personal pronouns
            r'^\s*(am i|do i have|is my|will my|can you tell me about my)\b',
            r'^\s*(what is my|when was my|how is my|where is my)\b'
        ]
        
        # Check if query matches self-inquiry patterns
        for pattern in self_inquiry_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _is_patient_context_relevant(self, query: str, intent: str) -> bool:
        """Determine if patient context is relevant for this specific query"""
        query_lower = query.lower().strip()
        
        # Always relevant for TBI-related queries
        if intent == "tbi_retrieval":
            return True
        
        # Check for patient self-inquiry keywords
        self_inquiry_patterns = [
            # Direct self-reference
            r'\b(my|mine|myself|me)\b',
            # Information requests about self
            r'\b(tell me about|what do you know about|show me|information about)\s+(me|my|myself)\b',
            r'\b(my (medical )?history|my (health )?record|my information|my profile|my data)\b',
            r'\b(what.*you.*know.*about.*me|tell.*about.*my.*condition|my.*diagnosis)\b',
            r'\b(my.*symptoms|my.*injury|my.*accident|my.*tbi|my.*treatment)\b',
            # Registration and personal info
            r'\b(when.*i.*(register|sign|join)|my.*registration|registered.*date)\b',
            r'\b(my.*account|my.*profile|about.*my.*case)\b',
            # Recovery and progress
            r'\b(my.*recovery|my.*progress|how.*am.*i.*doing)\b',
            # General self-reference medical
            r'\b(am.*i|do.*i.*have|is.*my|will.*my)\b'
        ]
        
        # Check if query matches self-inquiry patterns
        for pattern in self_inquiry_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Relevant for medical queries that might relate to patient's conditions
        if intent == "medical_general":
            # Check if query mentions symptoms that could be related to patient's history
            symptom_keywords = ['pain', 'headache', 'dizzy', 'tired', 'sleep', 'memory', 'mood', 'symptom']
            if any(keyword in query_lower for keyword in symptom_keywords):
                return True
        
        # Check for continuation questions that might reference patient data
        if intent == "continuation_response":
            # If the previous response included patient context, this might be relevant too
            if self.last_assistant_message and len(self.last_assistant_message) > 100:
                return True
        
        # Not relevant for general conversation or non-medical queries
        return False
    
    async def process_query(self, query: str, patient_id: str) -> Dict[str, Any]:
        """
        Main method to process patient queries with routing and context
        """
        start_time = datetime.now()
        original_query = query  # Store original for tracking
        
        try:
            # Load patient data if not already loaded or if different patient
            if (not self.current_patient_data or 
                self.current_patient_data.get('patient_id') != patient_id):
                
                if not await self.load_patient_data(patient_id):
                    return {
                        "success": False,
                        "error": f"Could not load data for patient {patient_id}",
                        "response": "I'm sorry, I couldn't access your medical records. Please verify your patient ID.",
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }
                
                # Send welcome message for new patients
                if patient_id not in self.welcomed_patients:
                    patient_name = self.current_patient_data.get('first_name', 'there')
                    welcome_message = self._generate_welcome_message(patient_name)
                    
                    # Mark this patient as welcomed
                    self.welcomed_patients.add(patient_id)
                    
                    # Store welcome message in conversation memory
                    self._update_conversation_memory("Initial connection", welcome_message)
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    return {
                        "success": True,
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "query": "Initial connection",
                        "intent_classified": "welcome_message",
                        "agent_used": "Welcome Service",
                        "response": welcome_message,
                        "processing_time": processing_time,
                        "guardrail_action": "allowed",
                        "is_welcome_message": True
                    }
            
            # First check if it's a patient self-inquiry - these should always be allowed
            if self._is_patient_self_inquiry(query):
                return {
                    'allow': True,
                    'action': 'allow',
                    'classification': {'category': 'patient_self_inquiry', 'confidence': 0.95},
                    'response': None
                }

            power_of_patients_keywords = ['power of patients', 'platform', 'sallie', 'who created you', 'who made you']
            if any(keyword in query.lower() for keyword in power_of_patients_keywords):
                return {
                    'allow': True,
                    'action': 'allow', 
                    'classification': {'category': 'platform_question', 'confidence': 0.9},
                    'response': None
                }

            # GUARDRAIL CHECK - New step!
            if self.guardrail_agent:
                try:
                    guardrail_result = await self.guardrail_agent.process_query(query)
                    
                    if not guardrail_result['allow']:
                        # Query blocked by guardrail - return redirect response
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        # Update conversation memory with redirect
                        self._update_conversation_memory(query, guardrail_result['response'])
                        
                        return {
                            "success": True,
                            "patient_id": patient_id,
                            "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                            "query": query,
                            "intent_classified": "non_medical_redirect",
                            "agent_used": "Guardrail Agent",
                            "response": guardrail_result['response'],
                            "processing_time": processing_time,
                            "guardrail_action": "redirected",
                            "guardrail_category": guardrail_result['classification'].get('category', 'unknown'),
                            "guardrail_confidence": guardrail_result['classification'].get('confidence', 0)
                        }
                        
                except Exception as e:
                    logger.warning(f"Guardrail check failed, proceeding with query: {e}")
                    # Continue processing if guardrail fails

            
            # QUERY PARAPHRASING - BEFORE CLASSIFICATION FOR BETTER ACCURACY!
            paraphrased_query = await self._paraphrase_query(query, "general")
            
            # Classify query intent using PARAPHRASED query for better accuracy
            intent = await self._classify_query_intent(paraphrased_query)
            
            # Check if this is a patient self-inquiry for routing (check both original and paraphrased)
            is_self_inquiry = self._is_patient_self_inquiry(query) or self._is_patient_self_inquiry(paraphrased_query)
            
            # Route to appropriate agent using PARAPHRASED query
            if intent == "crisis":
                agent_response = await self._handle_crisis(paraphrased_query)
                agent_used = "Crisis Support"
            elif intent == "continuation_response":
                agent_response = await self._handle_continuation_response(paraphrased_query)
                agent_used = "Continuation Handler"
            elif intent == "medical_general":
                agent_response = await self._call_medical_agent(paraphrased_query)
                agent_used = "MedPalm"
            elif intent == "tbi_retrieval":
                agent_response = await self._call_retrieval_agent(paraphrased_query)
                agent_used = "TBI Retrieval"
            elif is_self_inquiry:
                # Handle patient self-inquiry with basic response
                patient_name = self.current_patient_data.get('first_name', 'there')
                agent_response = f"I have your medical information here, {patient_name}. Let me share what I know about your case based on your profile."
                agent_used = "Patient Information"
            else:
                # Handle general conversation with context
                agent_response = f"I understand you're asking: {paraphrased_query}. Let me help you with that."
                agent_used = "General Conversation"
            
            # Store agent used for memory system
            self._last_agent_used = agent_used
            
            # Generate final contextual response using ORIGINAL query for context
            final_response = await self._generate_contextual_response(original_query, agent_response, intent)
            
            # Update conversational memory with ORIGINAL query
            self._update_conversation_memory(original_query, final_response)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "patient_id": patient_id,
                "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                "query": original_query,
                "paraphrased_query": paraphrased_query if paraphrased_query != original_query else None,
                "intent_classified": intent,
                "agent_used": agent_used,
                "response": final_response,
                "processing_time": processing_time,
                "guardrail_action": "allowed",
                "medical_disclaimer_added": self._is_medical_query(intent),
                "patient_context_used": self._is_patient_context_relevant(original_query, intent),
                "self_inquiry_detected": is_self_inquiry
            }
            
        except Exception as e:
            logger.exception(f"Error processing query: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again.",
                "processing_time": processing_time
            }
    
    async def get_classification_reasoning(self, query: str) -> Dict[str, str]:
        """Get detailed reasoning for query classification (for debugging/transparency)"""
        try:
            # Get recent conversation context for reasoning
            recent_history = self._format_recent_history(num_entries=3) if self.session_history else ""
            
            reasoning_prompt = f"""
You are a healthcare query classifier. Analyze the user's query and explain your classification reasoning.

{"RECENT CONVERSATION:" if recent_history else ""}
{recent_history}

USER QUERY: "{query}"

CLASSIFICATION OPTIONS:
1. "crisis" - Mental health crises, suicide ideation, self-harm, emergencies
2. "medical_general" - General medical questions (diabetes, heart disease, medications, etc.)
3. "tbi_retrieval" - TBI, concussion, brain injury, head trauma questions
4. "continuation_response" - Responses like "yes", "no", "sure" continuing previous conversation
5. "general_conversation" - Greetings, thanks, casual conversation

Please provide:
1. Your classification choice
2. Detailed reasoning for this classification
3. Key words/phrases that influenced your decision
4. Whether conversation context influenced the decision
5. Alternative classifications you considered and why you rejected them

Format your response as:
CLASSIFICATION: [your choice]
REASONING: [detailed explanation]
KEY_PHRASES: [relevant words/phrases from query]
CONTEXT_INFLUENCE: [how conversation context affected decision]
ALTERNATIVES: [other options considered]
"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    reasoning_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            reasoning_text = response.text
            
            # Parse the response
            lines = reasoning_text.split('\n')
            result = {
                'query': query,
                'full_reasoning': reasoning_text
            }
            
            for line in lines:
                if line.startswith('CLASSIFICATION:'):
                    result['classification'] = line.replace('CLASSIFICATION:', '').strip()
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.replace('REASONING:', '').strip()
                elif line.startswith('KEY_PHRASES:'):
                    result['key_phrases'] = line.replace('KEY_PHRASES:', '').strip()
                elif line.startswith('CONTEXT_INFLUENCE:'):
                    result['context_influence'] = line.replace('CONTEXT_INFLUENCE:', '').strip()
                elif line.startswith('ALTERNATIVES:'):
                    result['alternatives'] = line.replace('ALTERNATIVES:', '').strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting classification reasoning: {e}")
            return {
                'query': query,
                'error': str(e),
                'full_reasoning': 'Unable to generate reasoning due to error'
            }

    async def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session including conversation memory"""
        return {
            "current_patient": {
                "id": self.current_patient_data.get('patient_id'),
                "name": self.current_patient_data.get('first_name'),
                "type": self.current_patient_data.get('patient_type')
            },
            "session_length": len(self.session_history),
            "pending_questions": len(self.pending_questions),
            "agents_used": list(set([entry.get('agent_used', 'Unknown') for entry in self.session_history])),
            "total_processing_time": sum([entry.get('processing_time', 0) for entry in self.session_history if 'processing_time' in entry]),
            "last_assistant_message_preview": self.last_assistant_message[:100] + "..." if self.last_assistant_message else "None",
            "memory_buffer_status": "Full" if len(self.session_history) >= 10 else f"{len(self.session_history)}/10"
        }
    
    def clear_session(self):
        """Clear current session data including conversation memory"""
        self.session_history.clear()
        self.pending_questions.clear()
        self.last_assistant_message = ""
        self.welcomed_patients.clear()  # Clear welcome tracking
        logger.info("Session and conversation memory cleared")
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status for debugging"""
        return {
            "session_history_size": len(self.session_history),
            "pending_questions": self.pending_questions,
            "last_assistant_message": self.last_assistant_message[:150] + "..." if self.last_assistant_message else "",
            "memory_buffer_full": len(self.session_history) >= 10,
            "recent_exchanges": [
                {"query": entry['query'][:50] + "...", "agent": entry.get('agent_used', 'Unknown')}
                for entry in list(self.session_history)[-3:]
            ]
        }


# Terminal Testing Interface
async def terminal_interface():
    """Interactive terminal interface for testing"""
    print("=" * 60)
    print("PROFESSIONAL PATIENT AGENT - TERMINAL INTERFACE")
    print("=" * 60)
    
    # Initialize agent
    print("Initializing Patient Agent...")
    agent = ProfessionalPatientAgent()
    
    # Connect to database
    print("Connecting to database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database. Please check your configuration.")
        return
    
    print("✅ Patient Agent initialized successfully!")
    print("\nCommands:")
    print("- Enter your query normally")
    print("- Type 'switch <patient_id>' to change patient")
    print("- Type 'welcome' to see welcome message for current patient")
    print("- Type 'summary' to see session summary")
    print("- Type 'memory' to see conversation memory status")
    print("- Type 'reasoning <your query>' to see classification reasoning")
    print("- Type 'clear' to clear session")
    print("- Type 'quit' to exit")
    print("-" * 60)
    
    current_patient_id = None
    
    while True:
        try:
            # Get patient ID if not set
            if not current_patient_id:
                current_patient_id = input("\nEnter Patient ID to start: ").strip()
                if not current_patient_id:
                    continue
                
                # Test loading patient data
                print(f"Loading patient data for ID: {current_patient_id}...")
                if await agent.load_patient_data(current_patient_id):
                    patient_name = agent.current_patient_data.get('first_name', 'Unknown')
                    print(f"✅ Patient loaded: {patient_name} (ID: {current_patient_id})")
                else:
                    print("❌ Failed to load patient data. Please check the Patient ID.")
                    current_patient_id = None
                    continue
            
            # Get user input
            user_input = input(f"\n[Patient {current_patient_id}] Your query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'welcome':
                if current_patient_id and agent.current_patient_data:
                    patient_name = agent.current_patient_data.get('first_name', 'there')
                    welcome_msg = agent._generate_welcome_message(patient_name)
                    print(f"\n🎉 WELCOME MESSAGE:")
                    print(welcome_msg)
                else:
                    print("Please load a patient first")
                continue
                agent.clear_session()
                print("✅ Session cleared")
                continue
            elif user_input.lower() == 'summary':
                summary = await agent.get_session_summary()
                print("\n📊 SESSION SUMMARY:")
                print(json.dumps(summary, indent=2))
                continue
            elif user_input.lower() == 'memory':
                memory_status = agent.get_conversation_status()
                print("\n🧠 CONVERSATION MEMORY STATUS:")
                print(json.dumps(memory_status, indent=2))
                continue
            elif user_input.lower().startswith('test-guardrail '):
                test_query = user_input[15:].strip()
                if test_query and agent.guardrail_agent:
                    print("Testing guardrail...")
                    guardrail_result = await agent.guardrail_agent.process_query(test_query)
                    print(f"\n🛡️ GUARDRAIL TEST RESULT:")
                    print(f"Query: {test_query}")
                    print(f"Allow: {guardrail_result['allow']}")
                    print(f"Action: {guardrail_result['action']}")
                    print(f"Category: {guardrail_result['classification'].get('category', 'N/A')}")
                    print(f"Confidence: {guardrail_result['classification'].get('confidence', 0):.2f}")
                    if not guardrail_result['allow']:
                        print(f"Redirect Response: {guardrail_result['response']}")
                else:
                    print("Please provide a query after 'test-guardrail' or guardrail not available")
                continue
            elif user_input.lower().startswith('reasoning '):
                reasoning_query = user_input[10:].strip()
                if reasoning_query:
                    print("Analyzing classification reasoning...")
                    reasoning = await agent.get_classification_reasoning(reasoning_query)
                    print(f"\n🧠 CLASSIFICATION REASONING:")
                    print(f"Query: {reasoning.get('query', 'N/A')}")
                    print(f"Classification: {reasoning.get('classification', 'N/A')}")
                    print(f"Reasoning: {reasoning.get('reasoning', 'N/A')}")
                    print(f"Key Phrases: {reasoning.get('key_phrases', 'N/A')}")
                    print(f"Context Influence: {reasoning.get('context_influence', 'N/A')}")
                    print(f"Alternatives: {reasoning.get('alternatives', 'N/A')}")
                else:
                    print("Please provide a query after 'reasoning'")
                continue
            elif user_input.lower().startswith('switch '):
                new_patient_id = user_input[7:].strip()
                if new_patient_id:
                    current_patient_id = new_patient_id
                    print(f"Switching to patient: {current_patient_id}")
                    # Note: The welcome message will be automatically sent when processing the next query
                continue
            
            # Process query
            print("Processing query...")
            result = await agent.process_query(user_input, current_patient_id)
            
            if result["success"]:
                # Handle welcome message differently
                if result.get('is_welcome_message'):
                    print(f"\n🎉 {result['response']}")
                else:
                    print(f"{result['response']}")
                
                    # Show paraphrasing if it occurred
                    if result.get('paraphrased_query'):
                        print(f"\n🔄 Query paraphrased (before classification): '{user_input}' → '{result['paraphrased_query']}'")
                
                disclaimer_status = "✅ Medical disclaimer added" if result.get('medical_disclaimer_added', False) else ""
                print(f"\n📍 Intent: {result['intent_classified']} | Agent: {result['agent_used']} | Time: {result['processing_time']:.2f}s {disclaimer_status}")
            else:
                print(f"\n❌ Error: {result['error']}")
                print(f"Response: {result['response']}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run terminal interface
    asyncio.run(terminal_interface())