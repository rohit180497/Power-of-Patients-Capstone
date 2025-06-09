import os
import json
import logging
import asyncio
import psycopg2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv
from datetime import datetime
from collections import deque
from dataclasses import dataclass

# Import your existing specialized agents
from src.medpalm.medical_assistant import MedPalmAgent
from src.retrieval.cdcretrieval import CDCTBIRetriever

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalysis:
    """Structured analysis of user query with LLM insights"""
    original_query: str
    paraphrased_query: str
    intent: str
    confidence: float
    is_continuation: bool
    needs_patient_context: bool
    is_medical: bool
    urgency_level: str
    requires_guardrail_check: bool
    reasoning: str

@dataclass
class ConversationMemory:
    """Structured conversation memory for context management"""
    exchanges: deque
    current_patient_id: str
    patient_context: str
    last_agent_used: str
    pending_followup: Optional[str]
    
    def add_exchange(self, query: str, response: str, agent_used: str):
        self.exchanges.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "agent": agent_used
        })
        self.last_agent_used = agent_used
    
    def get_recent_context(self, num_exchanges: int = 3) -> str:
        """Get formatted recent conversation context"""
        if not self.exchanges:
            return ""
        
        recent = list(self.exchanges)[-num_exchanges:]
        context_parts = []
        for exchange in recent:
            context_parts.append(f"Patient: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response'][:150]}...")
        
        return "\n".join(context_parts)

class LLMGuardrailAgent:
    """Intelligent LLM-powered guardrail system with memory awareness"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def analyze_query_safety(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Analyze query safety with conversation context"""
        
        json_template = """{
    "allow": true/false,
    "confidence": 0.0-1.0,
    "category": "medical_query|patient_self_inquiry|continuation|non_medical|harmful",
    "redirect_message": "message if blocked",
    "reasoning": "detailed explanation"
}"""

        guardrail_prompt = f"""
You are a healthcare AI guardrail system. Analyze if this query should be processed by medical AI agents.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT QUERY: "{query}"

GUIDELINES:
✅ ALLOW - Medical questions, TBI inquiries, health concerns, patient self-inquiries, platform questions
✅ ALLOW - Follow-up questions related to previous medical discussions
✅ ALLOW - Requests for patient's own medical information
✅ ALLOW - Continuation responses (yes/no/maybe) to medical questions

❌ BLOCK - Requests for illegal activities, harmful content, non-medical personal advice
❌ BLOCK - Questions completely unrelated to health/medical topics
❌ BLOCK - Requests that could harm patient safety

ANALYSIS REQUIRED:
1. Is this query healthcare-related (directly or through conversation context)?
2. Is this a continuation of a medical conversation?
3. Is this requesting patient's own medical information?
4. Could answering this query cause harm?
5. Should this be redirected to healthcare professionals?

Respond in JSON format:
{json_template}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    guardrail_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            # Parse JSON response
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            try:
                result = json.loads(result_text)
                return {
                    "allow": result.get("allow", True),
                    "confidence": result.get("confidence", 0.5),
                    "category": result.get("category", "unknown"),
                    "redirect_message": result.get("redirect_message", ""),
                    "reasoning": result.get("reasoning", "")
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse guardrail JSON response: {result_text}")
                return {"allow": True, "confidence": 0.3, "category": "parse_error", "redirect_message": "", "reasoning": "Failed to parse response"}
        
        except Exception as e:
            logger.error(f"Guardrail analysis failed: {e}")
            return {"allow": True, "confidence": 0.1, "category": "error", "redirect_message": "", "reasoning": "Guardrail system error"}

class LLMQueryAnalyzer:
    """Intelligent query analysis using LLM with conversation memory"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def analyze_query(self, query: str, memory: ConversationMemory) -> QueryAnalysis:
        """Comprehensive query analysis with memory context"""
        
        # Get conversation context
        conversation_context = memory.get_recent_context(3)
        patient_name = "the patient"
        
        # Extract patient name if available
        if memory.patient_context:
            try:
                if "first_name" in memory.patient_context:
                    # Simple extraction - you might want to improve this
                    for line in memory.patient_context.split('\n'):
                        if 'Name:' in line:
                            patient_name = line.split('Name:')[1].strip()
                            break
            except:
                pass
        
        json_template = """{
    "paraphrased_query": "clear, explicit version of the query",
    "intent": "primary intent from options above",
    "confidence": 0.0-1.0,
    "is_continuation": true/false,
    "needs_patient_context": true/false,
    "is_medical": true/false,
    "urgency_level": "emergency/high/medium/low",
    "requires_guardrail_check": true/false,
    "reasoning": "detailed explanation of analysis decisions",
    "agent_routing": "which agent should handle this: crisis|medical|tbi|patient_info|general"
}"""

        analysis_prompt = f"""
You are an advanced healthcare query analyzer. Analyze this query in context of the ongoing conversation.

PATIENT NAME: {patient_name}
CONVERSATION CONTEXT:
{conversation_context}

ORIGINAL QUERY: "{query}"

ANALYSIS TASKS:
1. **QUERY PARAPHRASING**: Rewrite the query to be explicit and clear, resolving pronouns and references using conversation context
2. **INTENT CLASSIFICATION**: Determine the primary intent
3. **CONTEXT ANALYSIS**: Assess what context and information is needed
4. **ROUTING DECISION**: Determine which agent should handle this

INTENT OPTIONS:
- "crisis" - Mental health emergency, suicide ideation, immediate danger
- "medical_general" - General medical questions, medications, treatments
- "tbi_specific" - TBI, concussion, brain injury questions
- "patient_self_inquiry" - Patient asking about their own medical information
- "continuation" - Following up on previous conversation (yes/no/clarification)
- "general_conversation" - Greetings, thanks, platform questions

URGENCY LEVELS: "emergency", "high", "medium", "low"

Respond in JSON format:
{json_template}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    analysis_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            # Parse JSON response
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            try:
                result = json.loads(result_text)
                
                return QueryAnalysis(
                    original_query=query,
                    paraphrased_query=result.get("paraphrased_query", query),
                    intent=result.get("intent", "general_conversation"),
                    confidence=result.get("confidence", 0.5),
                    is_continuation=result.get("is_continuation", False),
                    needs_patient_context=result.get("needs_patient_context", False),
                    is_medical=result.get("is_medical", False),
                    urgency_level=result.get("urgency_level", "low"),
                    requires_guardrail_check=result.get("requires_guardrail_check", True),
                    reasoning=result.get("reasoning", "No reasoning provided")
                )
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse analysis JSON: {result_text}")
                # Fallback analysis
                return self._fallback_analysis(query)
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Simple fallback analysis when LLM fails"""
        
        query_lower = query.lower()
        
        # Emergency detection
        if any(word in query_lower for word in ['suicide', 'kill myself', 'emergency', 'crisis']):
            intent = "crisis"
            urgency = "emergency"
        elif any(word in query_lower for word in ['tbi', 'concussion', 'brain injury']):
            intent = "tbi_specific"
            urgency = "medium"
        elif any(word in query_lower for word in ['my', 'mine', 'about me']):
            intent = "patient_self_inquiry"
            urgency = "medium"
        else:
            intent = "general_conversation"
            urgency = "low"
        
        return QueryAnalysis(
            original_query=query,
            paraphrased_query=query,
            intent=intent,
            confidence=0.3,
            is_continuation=False,
            needs_patient_context=(intent == "patient_self_inquiry"),
            is_medical=(intent in ["medical_general", "tbi_specific", "crisis"]),
            urgency_level=urgency,
            requires_guardrail_check=True,
            reasoning="Fallback analysis due to LLM error"
        )

class ProfessionalPatientAgent:
    """
    Enhanced Professional Patient Agent with LLM-powered intelligence
    """
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None):
        """Initialize the Enhanced Patient Agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        # Database connection
        self.db_connection = None
        
        # Conversation Memory
        self.memory = ConversationMemory(
            exchanges=deque(maxlen=20),  # Keep last 20 exchanges
            current_patient_id="",
            patient_context="",
            last_agent_used="",
            pending_followup=None
        )
        
        # Patient data
        self.current_patient_data = {}
        self.welcomed_patients = set()
        
        # Initialize LLM and components
        self._initialize_llm_components()
        self._initialize_specialized_agents()
        
        logger.info("Enhanced Professional Patient Agent initialized")
    
    def _initialize_llm_components(self):
        """Initialize LLM-powered components"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Initialize LLM-powered components
                self.query_analyzer = LLMQueryAnalyzer(self.gemini_model)
                self.guardrail_agent = LLMGuardrailAgent(self.gemini_model)
                
                logger.info("LLM components initialized successfully")
            else:
                logger.error("Gemini API key not provided")
                raise ValueError("Gemini API key required")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM components: {e}")
            raise
    
    def _initialize_specialized_agents(self):
        """Initialize specialized medical agents"""
        try:
            # Initialize MedPalm Agent
            self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
            logger.info("MedPalm Agent initialized")
        except Exception as e:
            logger.warning(f"MedPalm Agent not available: {e}")
            self.medical_agent = None
        
        try:
            # Initialize TBI Retrieval Agent
            self.retrieval_agent = CDCTBIRetriever(
                pinecone_api_key=self.pinecone_api_key,
                index_name=os.getenv("PINECONE_INDEX2_NAME"),
                embedding_model=os.getenv("EMBEDDING_MODEL"),
                llm_provider="gemini"
            )
            logger.info("TBI Retrieval Agent initialized")
        except Exception as e:
            logger.warning(f"TBI Retrieval Agent not available: {e}")
            self.retrieval_agent = None
    
    async def connect_to_database(self, db_config: Dict[str, str] = None) -> bool:
        """Connect to database with improved error handling"""
        try:
            if db_config is None:
                db_config = {
                    'user': os.getenv("user"),
                    'password': os.getenv("password"),
                    'host': os.getenv("host"),
                    'port': os.getenv("port", "6543"),
                    'dbname': os.getenv("dbname")
                }
            
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration: {missing_keys}")
                return False
            
            loop = asyncio.get_event_loop()
            self.db_connection = await loop.run_in_executor(
                None, lambda: psycopg2.connect(**db_config)
            )
            
            logger.info("Database connected successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Database connection failed: {e}")
            return False
    
    async def load_patient_data(self, patient_id: str) -> bool:
        """Load comprehensive patient data"""
        try:
            # Check and reconnect database if needed
            if not self._check_db_connection():
                if not await self.connect_to_database():
                    return False
            
            query = """
            SELECT patient_id, first_name, user_type, registered_at, country, city,
                   patient_type, patient_sub_type, age, tbi_incident_date, injury_from,
                   head_hit_location, has_tbi_before, total_tbi, immediate_symptoms_resulting,
                   describe_event, worst_symptoms, symptom_json, sdoh_json, therapy_json
            FROM patient_summary
            WHERE patient_id = %s
            """
            
            loop = asyncio.get_event_loop()
            
            def execute_query():
                cursor = self.db_connection.cursor()
                cursor.execute(query, (patient_id,))
                result = cursor.fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    patient_data = dict(zip(columns, result))
                    cursor.close()
                    return patient_data
                else:
                    cursor.close()
                    return None
            
            patient_data = await loop.run_in_executor(None, execute_query)
            
            if not patient_data:
                logger.error(f"Patient not found: {patient_id}")
                return False
            
            self.current_patient_data = patient_data
            self.memory.current_patient_id = patient_id
            self.memory.patient_context = self._build_patient_context(patient_data)
            
            logger.info(f"Patient data loaded: {patient_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading patient data: {e}")
            return False
    
    def _check_db_connection(self) -> bool:
        """Check if database connection is alive"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
        except:
            return False
        return False
    
    def _build_patient_context(self, patient_data: Dict) -> str:
        """Build structured patient context"""
        context_parts = [
            "=== PATIENT INFORMATION ===",
            f"Patient ID: {patient_data.get('patient_id', 'N/A')}",
            f"Name: {patient_data.get('first_name', 'N/A')}",
            f"Age: {patient_data.get('age', 'N/A')}",
            f"Location: {patient_data.get('city', 'N/A')}, {patient_data.get('country', 'N/A')}",
            f"Patient Type: {patient_data.get('patient_type', 'N/A')}",
            "",
            "=== TBI HISTORY ===",
            f"Previous TBI: {patient_data.get('has_tbi_before', 'N/A')}",
            f"Total TBI Count: {patient_data.get('total_tbi', 'N/A')}",
            f"Incident Date: {patient_data.get('tbi_incident_date', 'N/A')}",
            f"Injury Source: {patient_data.get('injury_from', 'N/A')}",
            f"Head Hit Location: {patient_data.get('head_hit_location', 'N/A')}",
            f"Event Description: {patient_data.get('describe_event', 'N/A')}",
            "",
            "=== SYMPTOMS ===",
            f"Immediate Symptoms: {patient_data.get('immediate_symptoms_resulting', 'N/A')}",
            f"Worst Symptoms: {patient_data.get('worst_symptoms', 'N/A')}"
        ]
        
        # Add JSON data if available
        for json_field in ['symptom_json', 'sdoh_json', 'therapy_json']:
            if patient_data.get(json_field):
                try:
                    json_data = json.loads(patient_data[json_field]) if isinstance(patient_data[json_field], str) else patient_data[json_field]
                    context_parts.append(f"{json_field.replace('_', ' ').title()}: {json.dumps(json_data, indent=2)}")
                except:
                    context_parts.append(f"{json_field.replace('_', ' ').title()}: {patient_data[json_field]}")
        
        return "\n".join(context_parts)
    
    def _generate_welcome_message(self, patient_name: str) -> str:
        """Generate personalized welcome message"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            greeting = "Good Morning"
        elif 12 <= current_hour < 17:
            greeting = "Good Afternoon" 
        elif 17 <= current_hour < 22:
            greeting = "Good Evening"
        else:
            greeting = "Hello"
        
        return f"""Hello {patient_name}, {greeting}! 

I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. I'm here to help you with your medical questions, TBI information, and healthcare guidance.

How can I assist you with your health concerns today?"""
    
    async def _handle_crisis_query(self, analysis: QueryAnalysis) -> str:
        """Handle mental health crisis with immediate support"""
        return """
🚨 I'm very concerned about what you're sharing with me. Your life has value and there are people who want to help you.

**IMMEDIATE HELP:**
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741
• International: befrienders.org

**Please reach out to:**
• Emergency services (911) if in immediate danger
• Your healthcare provider
• A trusted friend or family member
• Local crisis center

You don't have to go through this alone. Professional counselors are available 24/7 to provide support and help you work through these feelings.

Is there someone you can call right now? Would you like help finding local mental health resources?
"""
    
    async def _handle_medical_query(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle general medical queries"""
        try:
            if self.medical_agent:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.medical_agent.process_query(analysis.paraphrased_query)
                )
                return {
                    "response": response,
                    "agent_used": "Medical Agent"
                }
            else:
                return {
                    "response": f"I understand you're asking about {analysis.paraphrased_query}. While I don't have access to the full medical database right now, I recommend consulting with your healthcare provider for specific medical advice.",
                    "agent_used": "Medical Agent (Limited)"
                }
        except Exception as e:
            logger.error(f"Medical agent error: {e}")
            return {
                "response": "I'm experiencing technical difficulties accessing medical information. Please consult your healthcare provider.",
                "agent_used": "Medical Agent (Error)"
            }
    
    async def _handle_tbi_query(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle TBI-specific queries with fallback to MedPalm"""
        primary_response = ""
        fallback_used = False
        
        try:
            # Try TBI Retrieval Agent first
            if self.retrieval_agent:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.retrieval_agent.ask_question(analysis.paraphrased_query, top_k=8)
                )
                primary_response = result['answer']
                
                # Check if response is insufficient (too short, generic, or indicates lack of info)
                insufficient_indicators = [
                    "I don't have specific information",
                    "limited information",
                    "not available in my database",
                    "recommend checking other sources",
                    len(primary_response.strip()) < 100,
                    "difficult with the information I currently have"
                ]
                
                is_insufficient = any(indicator in primary_response.lower() if isinstance(indicator, str) 
                                    else indicator for indicator in insufficient_indicators)
                
                if is_insufficient:
                    logger.info("TBI response insufficient, activating MedPalm fallback...")
                    # Fallback to MedPalm for better TBI information
                    if self.medical_agent:
                        try:
                            fallback_response = await loop.run_in_executor(
                                None, lambda: self.medical_agent.process_query(analysis.paraphrased_query)
                            )
                            primary_response = fallback_response
                            fallback_used = True
                            logger.info("✅ MedPalm fallback successful - better response generated")
                        except Exception as e:
                            logger.error(f"❌ MedPalm fallback failed: {e}")
                            # Keep original TBI response if fallback fails
                
            else:
                # If no TBI agent, go directly to MedPalm
                if self.medical_agent:
                    loop = asyncio.get_event_loop()
                    primary_response = await loop.run_in_executor(
                        None, lambda: self.medical_agent.process_query(analysis.paraphrased_query)
                    )
                    fallback_used = True
                else:
                    primary_response = f"I understand you're asking about TBI: {analysis.paraphrased_query}. While I don't have access to the TBI database right now, I recommend consulting with your neurologist or healthcare provider."
            
        except Exception as e:
            logger.error(f"TBI query handling error: {e}")
            primary_response = "I'm experiencing technical difficulties accessing TBI information. Please consult your healthcare provider."
        
        return {
            "response": primary_response,
            "fallback_used": fallback_used,
            "agent_used": "MedPalm (Fallback)" if fallback_used else "TBI Specialist"
        }
    
    async def _handle_patient_self_inquiry(self, analysis: QueryAnalysis) -> str:
        """Handle patient asking about their own information"""
        if not self.current_patient_data:
            return "I don't have access to your medical records right now. Please verify your patient ID."
        
        patient_name = self.current_patient_data.get('first_name', 'there')
        
        # Create a comprehensive response about their information
        basic_info_response = f"I have your medical information here, {patient_name}. Let me share what I know about your case based on your profile."
        
        return basic_info_response
    
    async def _generate_contextual_response(self, agent_result: Dict[str, Any], analysis: QueryAnalysis) -> str:
        """Generate final contextual response using LLM with strict context control"""
        
        # For crisis situations, return immediately
        if analysis.intent == "crisis":
            return agent_result["response"]
        
        # Get conversation context with explicit memory state
        conversation_history = self.memory.get_recent_context(3)
        has_conversation_history = len(self.memory.exchanges) > 0
        patient_context = self.memory.patient_context if analysis.needs_patient_context else ""
        patient_name = self.current_patient_data.get('first_name', 'there')
        
        # Medical disclaimer
        medical_disclaimer = ""
        if analysis.is_medical:
            medical_disclaimer = """

⚠️ **Medical Disclaimer:** I am an AI assistant providing general health information for educational purposes only. This information should not replace professional medical consultation. Please consult with your healthcare provider, doctor, or qualified medical professional for personalized medical advice, proper diagnosis, and treatment options specific to your condition."""
        
        # Build context-aware prompt with strict memory controls
        response_prompt = f"""
You are Sallie, a professional healthcare assistant created by Power of Patients. Generate a natural, helpful response to the patient's current query.

CRITICAL MEMORY RULES:
- CONVERSATION HISTORY EXISTS: {'YES' if has_conversation_history else 'NO'}
- If NO conversation history exists, DO NOT reference previous discussions
- If YES, only reference if directly relevant to current query
- NEVER say "as I mentioned before" or "as we discussed" unless you can see it in the conversation history below

PATIENT INFORMATION:
- Name: {patient_name}
- Current Query: "{analysis.original_query}"
- Paraphrased Query: "{analysis.paraphrased_query}"
- Intent: {analysis.intent}

{'CONVERSATION HISTORY (Recent exchanges):' if has_conversation_history else 'CONVERSATION HISTORY: None (This is our first interaction)'}
{conversation_history if has_conversation_history else 'No previous conversation exists.'}

SPECIALIST AGENT RESPONSE:
Agent Used: {agent_result.get('agent_used', 'Unknown')}
Response: {agent_result['response']}

{'PATIENT MEDICAL CONTEXT:' if analysis.needs_patient_context else ''}
{patient_context if analysis.needs_patient_context else ''}

RESPONSE GUIDELINES:
1. Address the patient by name ({patient_name}) when appropriate
2. Provide a helpful, empathetic response to their CURRENT query
3. Build upon the specialist agent's response with context and patient care
4. If patient has medical history relevant to the query, mention it supportively
5. ONLY reference previous conversations if they exist in the history above
6. For medical topics, emphasize consulting healthcare providers
7. End with an appropriate follow-up question or offer of help
8. Keep response natural and conversational
9. DO NOT fabricate or assume previous conversations that aren't shown above

STRICT CONTEXT RULES:
- If conversation history shows NO previous exchanges, treat this as a fresh conversation
- Only reference "earlier" or "before" if you can see it in the conversation history
- When in doubt, focus on the current query without referencing past discussions

Generate a natural, contextual response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    response_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            final_response = response.text.strip() + medical_disclaimer
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            # Fallback to agent response with disclaimer
            return agent_result["response"] + medical_disclaimer
    
    async def process_query(self, query: str, patient_id: str) -> Dict[str, Any]:
        """
        Main query processing with LLM-powered intelligence
        """
        start_time = datetime.now()
        
        try:
            # Load patient data if needed
            if (not self.current_patient_data or 
                self.current_patient_data.get('patient_id') != patient_id):
                
                if not await self.load_patient_data(patient_id):
                    return {
                        "success": False,
                        "error": f"Could not load patient data for {patient_id}",
                        "response": "I'm sorry, I couldn't access your medical records. Please verify your patient ID.",
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }
                
                # Send welcome message for new patients
                if patient_id not in self.welcomed_patients:
                    patient_name = self.current_patient_data.get('first_name', 'there')
                    welcome_message = self._generate_welcome_message(patient_name)
                    self.welcomed_patients.add(patient_id)
                    
                    # Add to memory
                    self.memory.add_exchange("Initial connection", welcome_message, "Welcome Service")
                    
                    return {
                        "success": True,
                        "patient_id": patient_id,
                        "patient_name": patient_name,
                        "query": "Initial connection",
                        "intent_classified": "welcome_message",
                        "agent_used": "Welcome Service",
                        "response": welcome_message,
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "is_welcome_message": True
                    }
            
            # Step 1: Analyze query with LLM intelligence
            logger.info(f"Analyzing query: {query[:50]}...")
            analysis = await self.query_analyzer.analyze_query(query, self.memory)
            
            # Step 2: Guardrail check if required
            if analysis.requires_guardrail_check:
                conversation_context = self.memory.get_recent_context(3)
                guardrail_result = await self.guardrail_agent.analyze_query_safety(
                    analysis.paraphrased_query, conversation_context
                )
                
                if not guardrail_result["allow"]:
                    # Query blocked by guardrail
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Add to memory
                    self.memory.add_exchange(query, guardrail_result["redirect_message"], "Guardrail Agent")
                    
                    return {
                        "success": True,
                        "patient_id": patient_id,
                        "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                        "query": query,
                        "paraphrased_query": analysis.paraphrased_query,
                        "intent_classified": "non_medical_redirect",
                        "agent_used": "Guardrail Agent",
                        "response": guardrail_result["redirect_message"],
                        "processing_time": processing_time,
                        "guardrail_blocked": True,
                        "guardrail_category": guardrail_result["category"]
                    }
            
            # Step 3: Route to appropriate agent based on analysis
            logger.info(f"Routing query with intent: {analysis.intent}")
            
            agent_result = {}
            
            if analysis.intent == "crisis":
                crisis_response = await self._handle_crisis_query(analysis)
                agent_result = {"response": crisis_response, "agent_used": "Crisis Support"}
            elif analysis.intent == "medical_general":
                agent_result = await self._handle_medical_query(analysis)
            elif analysis.intent == "tbi_specific":
                agent_result = await self._handle_tbi_query(analysis)
            elif analysis.intent == "patient_self_inquiry":
                self_inquiry_response = await self._handle_patient_self_inquiry(analysis)
                agent_result = {"response": self_inquiry_response, "agent_used": "Patient Information"}
            else:
                # General conversation
                general_response = f"I understand you're asking: {analysis.paraphrased_query}. How can I assist you with your health concerns?"
                agent_result = {"response": general_response, "agent_used": "General Conversation"}
            
            # Step 4: Generate final contextual response
            final_response = await self._generate_contextual_response(agent_result, analysis)
            
            # Step 5: Update conversation memory
            self.memory.add_exchange(query, final_response, agent_result["agent_used"])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                "query": query,
                "paraphrased_query": analysis.paraphrased_query if analysis.paraphrased_query != query else None,
                "intent_classified": analysis.intent,
                "agent_used": agent_result["agent_used"],
                "response": final_response,
                "processing_time": processing_time,
                "confidence": analysis.confidence,
                "urgency_level": analysis.urgency_level,
                "analysis_reasoning": analysis.reasoning,
                "fallback_used": agent_result.get("fallback_used", False)
            }
            
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again.",
                "processing_time": processing_time
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        return {
            "current_patient": {
                "id": self.memory.current_patient_id,
                "name": self.current_patient_data.get('first_name', 'Unknown')
            },
            "conversation_length": len(self.memory.exchanges),
            "recent_exchanges": list(self.memory.exchanges)[-3:],
            "last_agent_used": self.memory.last_agent_used,
            "pending_followup": self.memory.pending_followup,
            "memory_usage": f"{len(self.memory.exchanges)}/20"
        }
    
    def clear_session(self):
        """Clear session data"""
        self.memory.exchanges.clear()
        self.memory.last_agent_used = ""
        self.memory.pending_followup = None
        self.welcomed_patients.clear()
        logger.info("Session cleared")

# Terminal Testing Interface
async def terminal_interface():
    """Enhanced terminal interface for testing"""
    print("=" * 80)
    print("🏥 ENHANCED PROFESSIONAL PATIENT AGENT - TERMINAL INTERFACE")
    print("=" * 80)
    
    # Initialize agent
    print("Initializing Enhanced Patient Agent...")
    agent = ProfessionalPatientAgent()
    
    # Connect to database
    print("Connecting to database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database")
        return
    
    print("✅ Enhanced Patient Agent initialized successfully!")
    print("\n💡 Features:")
    print("   • LLM-powered query analysis and paraphrasing")
    print("   • Intelligent guardrail system with context awareness")
    print("   • Smart conversation memory management") 
    print("   • Advanced intent classification")
    print("   • Contextual response generation")
    
    print("\n📋 Commands:")
    print("   • Enter queries naturally - the agent will analyze and route intelligently")
    print("   • 'switch <patient_id>' - Change patient")
    print("   • 'summary' - View session summary")
    print("   • 'analyze <query>' - See detailed query analysis")
    print("   • 'clear' - Clear session")
    print("   • 'quit' - Exit")
    print("-" * 80)
    
    current_patient_id = None
    
    while True:
        try:
            # Get patient ID if not set
            if not current_patient_id:
                current_patient_id = input("\nEnter Patient ID: ").strip()
                if not current_patient_id:
                    continue
            
            # Get user input
            user_input = input(f"\n[Patient {current_patient_id}] Query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower().startswith('switch '):
                new_patient_id = user_input[7:].strip()
                if new_patient_id:
                    current_patient_id = new_patient_id
                    print(f"🔄 Switched to patient: {current_patient_id}")
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_session_summary()
                print("\n📊 SESSION SUMMARY:")
                print(json.dumps(summary, indent=2, default=str))
                continue
            elif user_input.lower().startswith('analyze '):
                analyze_query = user_input[8:].strip()
                if analyze_query:
                    print("🔍 Analyzing query...")
                    analysis = await agent.query_analyzer.analyze_query(analyze_query, agent.memory)
                    print(f"\n🧠 QUERY ANALYSIS:")
                    print(f"   Original: {analysis.original_query}")
                    print(f"   Paraphrased: {analysis.paraphrased_query}")
                    print(f"   Intent: {analysis.intent} (confidence: {analysis.confidence:.2f})")
                    print(f"   Is Medical: {analysis.is_medical}")
                    print(f"   Needs Context: {analysis.needs_patient_context}")
                    print(f"   Urgency: {analysis.urgency_level}")
                    print(f"   Reasoning: {analysis.reasoning}")
                continue
            elif user_input.lower() == 'clear':
                agent.clear_session()
                print("✅ Session cleared")
                continue
            
            # Process query
            print("🤖 Processing query...")
            result = await agent.process_query(user_input, current_patient_id)
            
            if result["success"]:
                print(f"\n💬 Sallie: {result['response']}")
                
                # Show analysis details
                print(f"\n📊 Analysis:")
                print(f"   Intent: {result['intent_classified']}")
                print(f"   Agent: {result['agent_used']}")
                if result.get('fallback_used'):
                    print(f"   🔄 Fallback: TBI → MedPalm used")
                print(f"   Time: {result['processing_time']:.2f}s")
                if result.get('paraphrased_query'):
                    print(f"   Paraphrased: '{result['paraphrased_query']}'")
                if result.get('confidence'):
                    print(f"   Confidence: {result['confidence']:.2f}")
                if result.get('urgency_level'):
                    print(f"   Urgency: {result['urgency_level']}")
                
            else:
                print(f"\n❌ Error: {result['error']}")
                print(f"Response: {result['response']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(terminal_interface())