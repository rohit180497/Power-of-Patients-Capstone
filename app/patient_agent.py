import os
import json
import logging
import asyncio
import psycopg2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Optional, Any, List, Tuple
from dotenv import load_dotenv
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from enum import Enum
import concurrent.futures

# Import your existing specialized agents
from src.medpalm.medical_assistant import MedPalmAgent
from src.retrieval.cdcretrieval import CDCTBIRetriever
from src.locator.facility_locator import HealthcareWellnessLocator

# OPTIMIZED: Set logger level to WARNING to reduce verbose output
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors

class QueryIntent(Enum):
    """Standardized query intents"""
    CRISIS = "crisis"
    TBI_RELATED = "tbi_related" 
    MEDICAL_GENERAL = "medical_general"
    PATIENT_SELF_INQUIRY = "patient_self_inquiry"
    LOCATION_SEARCH = "location_search"
    CONTINUATION = "continuation"
    GENERAL_CONVERSATION = "general_conversation"

class UrgencyLevel(Enum):
    """Standardized urgency levels"""
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class QueryAnalysis:
    """Comprehensive query analysis structure"""
    original_query: str
    paraphrased_query: str
    intent: QueryIntent
    confidence: float
    urgency: UrgencyLevel
    is_medical: bool
    requires_patient_context: bool
    agent_recommendations: List[str]
    medical_entities: List[str]
    reasoning: str

@dataclass
class AgentResponse:
    """Standardized agent response structure"""
    agent_name: str
    response: str
    success: bool
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    # New fields for JSON response handling
    structured_data: Optional[Dict[str, Any]] = None
    render_mode: Optional[str] = None

@dataclass
class ConversationMemory:
    """Enhanced conversation memory"""
    exchanges: deque
    current_patient_id: str
    patient_context: str
    session_start: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'session_start'):
            self.session_start = datetime.now()
    
    def add_exchange(self, query: str, response: str, intent: str, agents_used: List[str]):
        """Add exchange with metadata"""
        self.exchanges.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:150] + "..." if len(response) > 150 else response,  # OPTIMIZED: Truncate stored responses
            "intent": intent,
            "agents_used": agents_used
        })
    
    def get_conversation_context(self, max_exchanges: int = 3) -> str:
        """Get formatted conversation context"""
        if not self.exchanges:
            return ""
        
        recent = list(self.exchanges)[-max_exchanges:]
        context_parts = []
        
        for i, exchange in enumerate(recent, 1):
            context_parts.append(f"Exchange {i}:")
            context_parts.append(f"Patient: {exchange['query']}")
            context_parts.append(f"Assistant: {exchange['response']}")  # Already truncated in add_exchange
            context_parts.append("")
        
        return "\n".join(context_parts)

class IntelligentQueryProcessor:
    """Handles query paraphrasing and classification"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def process_and_classify_query(self, query: str, conversation_context: str = "", patient_context: str = "") -> QueryAnalysis:
        """Process query through paraphrasing and classification in one step"""
        
        processing_prompt = f"""
You are an expert medical AI system analyzer. Process this patient query through paraphrasing and classification.

CONVERSATION HISTORY:
{conversation_context if conversation_context else "No previous conversation"}

PATIENT MEDICAL CONTEXT:
{patient_context[:500] if patient_context else "Patient context not available"}

ORIGINAL PATIENT QUERY: "{query}"

TASK 1 - QUERY PARAPHRASING:
Create a clear, explicit version of the query that:
- Resolves pronouns and ambiguous references using context
- Makes implicit requests explicit
- Maintains the original medical intent
- Improves clarity for agent processing

TASK 2 - INTENT CLASSIFICATION:
Classify the paraphrased query into one of these categories:
1. "crisis" - Mental health emergency, suicide ideation, immediate danger
2. "tbi_related" - TBI, concussion, brain injury, post-concussion symptoms
3. "medical_general" - Other medical conditions, symptoms, treatments
4. "patient_self_inquiry" - Questions about their own medical records/history
5. "location_search" - Finding healthcare facilities, wellness centers, therapy centers
6. "continuation" - Follow-up responses to previous medical discussion
7. "general_conversation" - Greetings, thanks, non-medical chat

TASK 3 - AGENT RECOMMENDATION:
Based on the intent, recommend which agents should process this query:
- "medpalm" - For general medical questions and advice
- "tbi_retrieval" - For TBI-specific information and research
- "location_search" - For finding healthcare facilities and centers
- Multiple agents can be recommended for comprehensive responses

MEDICAL ENTITY DETECTION:
Identify medical terms, conditions, symptoms, treatments mentioned in the query.

RESPONSE FORMAT (JSON):
{{
    "paraphrased_query": "Clear, explicit version of the query",
    "intent": "one of the 7 categories above",
    "confidence": 0.0-1.0,
    "urgency": "emergency/high/medium/low",
    "is_medical": true/false,
    "requires_patient_context": true/false,
    "agent_recommendations": ["list", "of", "recommended", "agents"],
    "medical_entities": ["list", "of", "medical", "terms"],
    "reasoning": "Detailed explanation of classification and agent selection decisions"
}}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    processing_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            result_text = response.text.strip()
            # Clean JSON response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            result = json.loads(result_text)
            
            # Map string values to enums
            intent_mapping = {
                "crisis": QueryIntent.CRISIS,
                "tbi_related": QueryIntent.TBI_RELATED,
                "medical_general": QueryIntent.MEDICAL_GENERAL,
                "patient_self_inquiry": QueryIntent.PATIENT_SELF_INQUIRY,
                "location_search": QueryIntent.LOCATION_SEARCH,
                "continuation": QueryIntent.CONTINUATION,
                "general_conversation": QueryIntent.GENERAL_CONVERSATION
            }
            
            urgency_mapping = {
                "emergency": UrgencyLevel.EMERGENCY,
                "high": UrgencyLevel.HIGH,
                "medium": UrgencyLevel.MEDIUM,
                "low": UrgencyLevel.LOW
            }
            
            return QueryAnalysis(
                original_query=query,
                paraphrased_query=result.get("paraphrased_query", query),
                intent=intent_mapping.get(result.get("intent"), QueryIntent.GENERAL_CONVERSATION),
                confidence=result.get("confidence", 0.5),
                urgency=urgency_mapping.get(result.get("urgency"), UrgencyLevel.LOW),
                is_medical=result.get("is_medical", False),
                requires_patient_context=result.get("requires_patient_context", False),
                agent_recommendations=result.get("agent_recommendations", []),
                medical_entities=result.get("medical_entities", []),
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Simple fallback that still goes through LLM later
            return QueryAnalysis(
                original_query=query,
                paraphrased_query=query,
                intent=QueryIntent.GENERAL_CONVERSATION,
                confidence=0.3,
                urgency=UrgencyLevel.LOW,
                is_medical=False,
                requires_patient_context=False,
                agent_recommendations=["medpalm"],
                medical_entities=[],
                reasoning="Fallback due to processing error"
            )

class MultiAgentOrchestrator:
    """Orchestrates multiple agents based on intelligent recommendations"""
    
    def __init__(self, medical_agent, tbi_agent, locator_agent):
        self.agents = {
            "medpalm": medical_agent,
            "tbi_retrieval": tbi_agent,
            "location_search": locator_agent
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    async def execute_agents(self, paraphrased_query: str, agent_recommendations: List[str]) -> List[AgentResponse]:
        """Execute recommended agents in parallel"""
        tasks = []
        
        for agent_name in agent_recommendations:
            if agent_name in self.agents and self.agents[agent_name] is not None:
                if agent_name == "medpalm":
                    tasks.append(self._execute_medpalm(paraphrased_query))
                elif agent_name == "tbi_retrieval":
                    tasks.append(self._execute_tbi_retrieval(paraphrased_query))
                elif agent_name == "location_search":
                    tasks.append(self._execute_location_search(paraphrased_query))
        
        if not tasks:
            # If no valid agents, default to medical agent
            if self.agents["medpalm"]:
                tasks.append(self._execute_medpalm(paraphrased_query))
        
        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        valid_responses = []
        for response in responses:
            if isinstance(response, AgentResponse):
                valid_responses.append(response)
            elif isinstance(response, Exception):
                logger.error(f"Agent execution failed: {response}")
        
        return valid_responses
    
    async def _execute_medpalm(self, query: str) -> AgentResponse:
        """Execute MedPalm agent"""
        start_time = datetime.now()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.agents["medpalm"].process_query(query)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name="MedPalm Medical Assistant",
                response=response,
                success=True,
                confidence=0.85,
                processing_time=processing_time,
                metadata={"source": "medical_knowledge_base"}
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                agent_name="MedPalm Medical Assistant",
                response="",
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _execute_tbi_retrieval(self, query: str) -> AgentResponse:
        """Execute TBI retrieval agent"""
        start_time = datetime.now()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.agents["tbi_retrieval"].ask_question(query, top_k=8)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name="TBI Research Specialist",
                response=result.get('answer', ''),
                success=True,
                confidence=0.75,
                processing_time=processing_time,
                metadata={"source": "tbi_research_database", "sources": result.get('sources', [])}
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                agent_name="TBI Research Specialist",
                response="",
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _execute_location_search(self, query: str) -> AgentResponse:
        """Execute location search agent with JSON support - OPTIMIZED LOGGING"""
        start_time = datetime.now()
        try:
            result = await self.agents["location_search"].find_facilities(query)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # OPTIMIZED: Minimal logging only for errors
            if not result.get("success"):
                logger.error(f"Location search failed: {result.get('error', 'Unknown error')}")
            
            if result.get("success"):
                return AgentResponse(
                    agent_name="Healthcare & Wellness Locator",
                    response=result.get("response", ""),
                    success=True,
                    confidence=result.get("confidence", 0.8),
                    processing_time=processing_time,
                    metadata={
                        "facilities_found": result.get("total_results", 0),
                        "location": result.get("location", ""),
                        "categories": result.get("categories_searched", [])
                    },
                    # Store structured data and render mode
                    structured_data=result.get("structured_data"),
                    render_mode=result.get("render_mode", "structured")
                )
            else:
                return AgentResponse(
                    agent_name="Healthcare & Wellness Locator",
                    response="",
                    success=False,
                    confidence=0.0,
                    processing_time=processing_time,
                    error_message=result.get("error", "Location search failed"),
                    structured_data=result.get("structured_data"),
                    render_mode=result.get("render_mode", "error")
                )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Location search execution failed: {e}")
            return AgentResponse(
                agent_name="Healthcare & Wellness Locator",
                response="",
                success=False,
                confidence=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )

class IntelligentResponseSynthesizer:
    """LLM-powered response synthesis - handles all scenarios intelligently"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def synthesize_comprehensive_response(
        self, 
        analysis: QueryAnalysis,
        agent_responses: List[AgentResponse],
        patient_context: str = "",
        conversation_context: str = ""
    ) -> str:
        """Synthesize comprehensive response using LLM intelligence - no fallbacks"""
        
        # Prepare agent information for LLM
        agent_info = self._prepare_agent_information(agent_responses)
        
        synthesis_prompt = f"""
You are Sallie, a professional and empathetic healthcare assistant created by Power of Patients. 
You must provide a comprehensive, helpful response to the patient's query using all available information.

PATIENT'S ORIGINAL QUERY: "{analysis.original_query}"
PARAPHRASED QUERY: "{analysis.paraphrased_query}"
QUERY INTENT: {analysis.intent.value}
URGENCY LEVEL: {analysis.urgency.value}
MEDICAL ENTITIES IDENTIFIED: {', '.join(analysis.medical_entities) if analysis.medical_entities else 'None'}

CONVERSATION HISTORY:
{conversation_context if conversation_context else "No previous conversation"}

PATIENT MEDICAL PROFILE:
{patient_context if analysis.requires_patient_context and patient_context else "Patient context not needed for this response"}

AGENT RESPONSES:
{agent_info}

CRITICAL INSTRUCTIONS:
1. **ALWAYS provide a concise, helpful response** - Never say you cannot help or provide partial information
2. **Handle ALL scenarios intelligently**:
   - If agents succeeded: Synthesize their responses comprehensively
   - If agents failed: Use your medical knowledge to provide helpful guidance
   - If no agents ran: Still provide valuable medical information and guidance
   - If location search failed: Give general advice on finding healthcare facilities
   - If this is a crisis: Provide immediate crisis resources and support

3. **Response Quality Standards**:
   - Be comprehensive and thorough
   - Use clear, patient-friendly language
   - Structure information logically
   - Provide actionable next steps
   - Be empathetic and supportive
   - Include relevant medical disclaimer when appropriate

4. **For Different Query Types**:
   - **Medical queries**: Provide detailed medical information, symptoms, treatments, when to seek care
   - **Location searches**: List specific facilities if found, or guide on how to find them
   - **TBI queries**: Comprehensive TBI information, management, recovery guidance
   - **Crisis situations**: Immediate resources, safety planning, professional help contacts
   - **Patient inquiries**: Use their medical history to provide personalized guidance

5. **Never use phrases like**:
   - "I'm sorry, I cannot help"
   - "I don't have access to"
   - "Unfortunately, I cannot provide"
   - "I'm experiencing technical difficulties"

Instead, ALWAYS find a way to be helpful using available information and your medical knowledge.

Generate a comprehensive, professional healthcare response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    synthesis_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            final_response = response.text.strip()
            
            # Add medical disclaimer for medical queries
            if analysis.is_medical:
                final_response = self._add_medical_disclaimer(final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Even in error case, let LLM handle it
            return await self._emergency_llm_response(analysis, agent_responses)
    
    def _prepare_agent_information(self, agent_responses: List[AgentResponse]) -> str:
        """Prepare agent response information for LLM - OPTIMIZED: Truncated responses"""
        if not agent_responses:
            return "No specialized agents were executed for this query."
        
        agent_info = []
        for response in agent_responses:
            status = "✅ SUCCESS" if response.success else "❌ FAILED"
            # OPTIMIZED: Truncate long responses for synthesis
            response_text = response.response[:500] + "..." if len(response.response) > 500 else response.response
            agent_info.append(f"""
{status} {response.agent_name} (Confidence: {response.confidence:.2f})
Response: {response_text if response.success else f"Error: {response.error_message}"}""")
        
        return "\n".join(agent_info)
    
    async def _emergency_llm_response(self, analysis: QueryAnalysis, agent_responses: List[AgentResponse]) -> str:
        """Emergency LLM response when synthesis fails"""
        emergency_prompt = f"""
You are Sallie, a healthcare assistant. There was a technical issue, but you must still help the patient.

Patient asked: "{analysis.original_query}"
Query type: {analysis.intent.value}
Medical entities: {', '.join(analysis.medical_entities)}

Provide helpful medical guidance for this query using your knowledge. Be comprehensive and supportive.
Always include appropriate medical disclaimer for health-related responses.
"""
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(emergency_prompt, safety_settings=self.safety_settings)
            )
            
            result = response.text.strip()
            if analysis.is_medical:
                result = self._add_medical_disclaimer(result)
            return result
            
        except Exception as e:
            logger.error(f"Emergency response failed: {e}")
            return "I'm here to help with your healthcare questions. While I'm experiencing some technical issues right now, please don't hesitate to reach out to your healthcare provider for immediate assistance with any urgent medical concerns."
    
    def _add_medical_disclaimer(self, response: str) -> str:
        """Add medical disclaimer"""
        disclaimer = "\n\n⚠️ **Medical Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Please consult with your healthcare provider for personalized medical guidance."
        return response + disclaimer

class EnhancedProfessionalPatientAgent:
    """Restructured Professional Patient Agent with JSON response handling - OPTIMIZED VERSION"""
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None, google_places_api_key: str = None):
        """Initialize the Enhanced Patient Agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.google_places_api_key = google_places_api_key or os.getenv("GOOGLE_PLACES_API_KEY")
        
        # Database connection
        self.db_connection = None
        
        # Enhanced conversation memory
        self.memory = ConversationMemory(
            exchanges=deque(maxlen=20),
            current_patient_id="",
            patient_context="",
            session_start=datetime.now()
        )
        
        # Patient data
        self.current_patient_data = {}
        self.welcomed_patients = set()
        
        # Initialize system components
        self._initialize_system()
        
        # OPTIMIZED: Minimal startup logging
        print("✅ Enhanced Patient Agent initialized")
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize Gemini
            if not self.gemini_api_key:
                raise ValueError("Gemini API key required")
            
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize specialized agents
            self.medical_agent = None
            self.tbi_agent = None
            self.locator_agent = None
            
            # Initialize Medical Agent
            try:
                self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
            except Exception as e:
                logger.error(f"Medical Agent not available: {e}")
            
            # Initialize TBI Agent
            try:
                self.tbi_agent = CDCTBIRetriever(
                    pinecone_api_key=self.pinecone_api_key,
                    index_name=os.getenv("PINECONE_INDEX2_NAME"),
                    embedding_model=os.getenv("EMBEDDING_MODEL"),
                    llm_provider="gemini"
                )
            except Exception as e:
                logger.error(f"TBI Agent not available: {e}")
            
            # Initialize Locator Agent
            try:
                if self.google_places_api_key:
                    self.locator_agent = HealthcareWellnessLocator(
                        gemini_api_key=self.gemini_api_key,
                        google_places_api_key=self.google_places_api_key
                    )
                else:
                    logger.error("Google Places API key not provided")
            except Exception as e:
                logger.error(f"Locator Agent not available: {e}")
            
            # Initialize core system components
            self.query_processor = IntelligentQueryProcessor(self.gemini_model)
            self.orchestrator = MultiAgentOrchestrator(self.medical_agent, self.tbi_agent, self.locator_agent)
            self.synthesizer = IntelligentResponseSynthesizer(self.gemini_model)
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def connect_to_database(self, db_config: Dict[str, str] = None) -> bool:
        """Connect to database"""
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
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def load_patient_data(self, patient_id: str) -> bool:
        """Load patient data"""
        try:
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            return False
    
    def _check_db_connection(self) -> bool:
        """Check database connection"""
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
        """Build comprehensive patient context"""
        context_parts = [
            f"Patient: {patient_data.get('first_name', 'Unknown')} (ID: {patient_data.get('patient_id')})",
            f"Age: {patient_data.get('age', 'N/A')}, Location: {patient_data.get('city', 'N/A')}, {patient_data.get('country', 'N/A')}",
            f"Patient Type: {patient_data.get('patient_type', 'N/A')}",
            "",
            "TBI/INJURY HISTORY:",
            f"- Previous TBI: {patient_data.get('has_tbi_before', 'N/A')}",
            f"- Total TBI incidents: {patient_data.get('total_tbi', 'N/A')}",
            f"- Recent incident: {patient_data.get('tbi_incident_date', 'N/A')}",
            f"- Injury source: {patient_data.get('injury_from', 'N/A')}",
            f"- Impact location: {patient_data.get('head_hit_location', 'N/A')}",
            "",
            "SYMPTOMS:",
            f"- Immediate symptoms: {patient_data.get('immediate_symptoms_resulting', 'N/A')}",
            f"- Worst symptoms: {patient_data.get('worst_symptoms', 'N/A')}"
        ]
        
        # Add JSON data if available
        for field, title in [('symptom_json', 'Current Symptoms'), ('therapy_json', 'Treatment History')]:
            if patient_data.get(field):
                try:
                    json_data = json.loads(patient_data[field]) if isinstance(patient_data[field], str) else patient_data[field]
                    context_parts.append(f"\n{title.upper()}:")
                    if isinstance(json_data, dict):
                        for key, value in json_data.items():
                            context_parts.append(f"- {key}: {value}")
                    else:
                        context_parts.append(f"- {json_data}")
                except:
                    pass
        
        return "\n".join(context_parts)
    
    def _generate_welcome_message(self, patient_name: str) -> str:
        """Generate personalized welcome"""
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

I'm Sallie, your healthcare assistant from Power of Patients. I'm here to provide comprehensive support for your medical questions and health concerns.

I can help you with:
• **Medical information** - symptoms, conditions, treatments
• **TBI and concussion guidance** - recovery, management, symptoms
• **Healthcare navigation** - finding facilities and resources
• **Personalized advice** - based on your medical history

What would you like to know about your health today?"""
    
    def _should_return_json_directly(self, analysis: QueryAnalysis, agent_responses: List[AgentResponse]) -> bool:
        """Determine if we should return JSON directly instead of synthesizing"""
        
        # Check if this is a location search with locator agent response
        if analysis.intent == QueryIntent.LOCATION_SEARCH:
            # Find locator agent response
            locator_response = None
            for response in agent_responses:
                if "Locator" in response.agent_name and response.success:
                    locator_response = response
                    break
            
            if locator_response:
                return True
        
        return False
    
    def _create_json_response(self, analysis: QueryAnalysis, agent_responses: List[AgentResponse], patient_id: str, start_time: datetime) -> Dict[str, Any]:
        """Create JSON response for direct frontend rendering - OPTIMIZED: No Duplication"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Find the locator agent response
        locator_response = None
        for response in agent_responses:
            if "Locator" in response.agent_name:
                locator_response = response
                break
        
        if not locator_response:
            # Fallback if no locator response found
            return {
                "success": False,
                "error": "No location data available",
                "response_format": "json"
            }
        
        # Extract structured data from the locator response
        structured_data = locator_response.structured_data
        
        # If structured_data is not available, create it from available information  
        if not structured_data:
            facilities_count = locator_response.metadata.get('facilities_found', 0) if locator_response.metadata else 0
            location = locator_response.metadata.get('location', '') if locator_response.metadata else ''
            
            # Try to extract facility data from the text response
            facilities = self._extract_facilities_from_text(locator_response.response)
            
            # Create structured data
            structured_data = {
                "header": {
                    "title": f"Found {facilities_count or len(facilities)} facilities near {location}",
                    "subtitle": "Location search results",
                    "categories": [{"name": "Wellness Centers", "icon": "🌿"}]
                },
                "search_metadata": {
                    "original_query": analysis.original_query,
                    "total_found": facilities_count or len(facilities),
                    "location": location,
                    "showing_count": len(facilities)
                },
                "facilities": facilities,
                "message": "Location search completed successfully",
                "rendering_hints": {
                    "map_view_available": True,
                    "list_view_default": True
                }
            }
        
        # OPTIMIZED: Create clean JSON response without duplication
        return {
            "success": True,
            "patient_id": patient_id,
            "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
            "query": analysis.original_query,
            "intent_classified": analysis.intent.value,
            "urgency_level": analysis.urgency.value,
            "processing_time": processing_time,
            "confidence": analysis.confidence,
            
            # JSON Response specific fields  
            "response_format": "json",
            "render_mode": locator_response.render_mode or "structured",
            "structured_data": structured_data,
            
            # Essential metadata only (no duplication)
            "facilities_found": locator_response.metadata.get("facilities_found", 0) if locator_response.metadata else 0,
            "search_location": locator_response.metadata.get("location", "") if locator_response.metadata else "",
            
            # For frontend identification
            "is_location_json": True
        }
    
    def _extract_facilities_from_text(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract facility data from text response and convert to structured format"""
        
        facilities = []
        
        try:
            # Split response by facility entries (looking for numbered entries)
            lines = response_text.split('\n')
            current_facility = {}
            
            for line in lines:
                line = line.strip()
                
                # Detect facility name (usually starts with ** and a number)
                if line.startswith('**') and ('.' in line):
                    # Save previous facility if exists
                    if current_facility.get('name'):
                        facilities.append(current_facility)
                    
                    # Start new facility
                    current_facility = {}
                    # Extract name (remove markdown and numbering)
                    name = line.replace('**', '').strip()
                    # Remove number prefix (like "1. ")
                    if '. ' in name:
                        name = name.split('. ', 1)[1] if len(name.split('. ', 1)) > 1 else name
                    current_facility['name'] = name
                    current_facility['id'] = f"facility_{len(facilities)}"
                
                # Extract address (usually starts with 📍)
                elif '📍' in line:
                    address = line.replace('📍', '').strip()
                    current_facility['location'] = {
                        'address': address,
                        'distance_display': None
                    }
                
                # Extract reviews and ratings (usually has 👥 and ⭐)
                elif '👥' in line and '⭐' in line:
                    # Extract review count and rating
                    parts = line.split('|')
                    if len(parts) >= 2:
                        # Review count
                        review_part = parts[0].replace('👥', '').strip()
                        if 'reviews' in review_part:
                            try:
                                review_count = int(review_part.split()[0])
                                current_facility['reviews'] = {'review_count': review_count}
                            except:
                                pass
                        
                        # Rating
                        rating_part = parts[1].strip()
                        if '/5' in rating_part:
                            try:
                                rating_str = rating_part.split('⭐')[-1].strip()
                                rating = float(rating_str.split('/5')[0].strip())
                                if 'reviews' not in current_facility:
                                    current_facility['reviews'] = {}
                                current_facility['reviews']['rating'] = rating
                                current_facility['reviews']['stars_display'] = '⭐' * int(rating)
                            except:
                                pass
                
                # Extract distance (usually starts with 📏)
                elif '📏' in line:
                    distance = line.replace('📏', '').strip()
                    if 'location' not in current_facility:
                        current_facility['location'] = {}
                    current_facility['location']['distance_display'] = distance
                    
                    # Extract numeric distance
                    try:
                        distance_miles = float(distance.split()[0])
                        current_facility['location']['distance_miles'] = distance_miles
                    except:
                        pass
                
                # Extract maps URL (usually starts with 🗺️)
                elif '🗺️' in line and 'place_id' in line:
                    # Extract place_id from the URL
                    import re
                    place_id_match = re.search(r'place_id:([^)]+)', line)
                    if place_id_match:
                        place_id = place_id_match.group(1)
                        current_facility['actions'] = {
                            'maps_url': f"https://www.google.com/maps/place/?q=place_id:{place_id}",
                            'place_id': place_id
                        }
            
            # Don't forget the last facility
            if current_facility.get('name'):
                facilities.append(current_facility)
            
            # Add category and contact info to each facility
            for facility in facilities:
                if 'category' not in facility:
                    facility['category'] = {
                        'name': 'Wellness Center',
                        'icon': '🌿'
                    }
                if 'contact' not in facility:
                    facility['contact'] = {}
                if 'business_info' not in facility:
                    facility['business_info'] = {'status': 'OPERATIONAL'}
            
        except Exception as e:
            logger.error(f"Failed to extract facilities from text: {e}")
        
        return facilities
    
    async def process_query(self, query: str, patient_id: str) -> Dict[str, Any]:
        """
        MAIN WORKFLOW: User query -> LLM paraphrase -> Classify -> Agents -> Smart Response (JSON or Synthesis)
        OPTIMIZED: Minimal logging, no duplication
        """
        start_time = datetime.now()
        
        try:
            # Ensure patient data is loaded
            if (not self.current_patient_data or 
                self.current_patient_data.get('patient_id') != patient_id):
                await self._setup_patient_session(patient_id)
                
                if not self.current_patient_data:
                    # Let LLM handle missing patient data
                    analysis = QueryAnalysis(
                        original_query=query,
                        paraphrased_query=query,
                        intent=QueryIntent.GENERAL_CONVERSATION,
                        confidence=0.5,
                        urgency=UrgencyLevel.LOW,
                        is_medical=False,
                        requires_patient_context=False,
                        agent_recommendations=["medpalm"],
                        medical_entities=[],
                        reasoning="Patient data unavailable"
                    )
                    
                    response = await self.synthesizer.synthesize_comprehensive_response(
                        analysis, [], "", ""
                    )
                    
                    return {
                        "success": True,
                        "patient_id": patient_id,
                        "query": query,
                        "response": response,
                        "response_format": "text",
                        "processing_time": (datetime.now() - start_time).total_seconds(),
                        "agents_used": ["LLM Synthesis Only"]
                    }
                
                # Send welcome for new sessions
                if patient_id not in self.welcomed_patients:
                    return self._create_welcome_response(patient_id, start_time)
            
            # STEP 1: LLM Paraphrase + Classify Query
            conversation_context = self.memory.get_conversation_context(3)
            patient_context = self.memory.patient_context if self.memory.current_patient_id == patient_id else ""
            
            analysis = await self.query_processor.process_and_classify_query(
                query, conversation_context, patient_context
            )
            
            # STEP 2: Execute Recommended Agents in Parallel
            agent_responses = await self.orchestrator.execute_agents(
                analysis.paraphrased_query, 
                analysis.agent_recommendations
            )
            
            # STEP 3: Decide Response Format (JSON vs Synthesized)
            if self._should_return_json_directly(analysis, agent_responses):
                # Return structured JSON response directly
                json_response = self._create_json_response(analysis, agent_responses, patient_id, start_time)
                
                # Still update memory with simplified text response for context
                simple_response = f"Found location results for: {analysis.original_query}"
                agents_used = [r.agent_name for r in agent_responses]
                self.memory.add_exchange(query, simple_response, analysis.intent.value, agents_used)
                
                return json_response
            
            else:
                # STEP 4: LLM-Powered Response Synthesis (Traditional Path)
                final_response = await self.synthesizer.synthesize_comprehensive_response(
                    analysis,
                    agent_responses,
                    patient_context if analysis.requires_patient_context else "",
                    conversation_context
                )
                
                # Update Memory
                agents_used = [r.agent_name for r in agent_responses] if agent_responses else ["LLM Knowledge Base"]
                self.memory.add_exchange(query, final_response, analysis.intent.value, agents_used)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "success": True,
                    "patient_id": patient_id,
                    "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                    "query": query,
                    "intent_classified": analysis.intent.value,
                    "urgency_level": analysis.urgency.value,
                    "agents_used": agents_used,
                    "response": final_response,
                    "response_format": "text",  # Indicates text synthesis
                    "processing_time": processing_time,
                    "confidence": analysis.confidence,
                    "medical_entities": analysis.medical_entities
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Even errors go through LLM
            emergency_response = await self.synthesizer.synthesize_comprehensive_response(
                QueryAnalysis(
                    original_query=query,
                    paraphrased_query=query,
                    intent=QueryIntent.GENERAL_CONVERSATION,
                    confidence=0.3,
                    urgency=UrgencyLevel.LOW,
                    is_medical=True,
                    requires_patient_context=False,
                    agent_recommendations=[],
                    medical_entities=[],
                    reasoning="Error handling"
                ),
                [],
                "",
                ""
            )
            
            return {
                "success": True,  # Still successful because LLM handled it
                "patient_id": patient_id,
                "query": query,
                "response": emergency_response,
                "response_format": "text",
                "processing_time": processing_time,
                "agents_used": ["Emergency LLM Response"]
            }
    
    async def _setup_patient_session(self, patient_id: str):
        """Setup patient session"""
        if self.memory.current_patient_id != patient_id:
            self.memory.exchanges.clear()
            self.welcomed_patients.discard(patient_id)
        
        await self.load_patient_data(patient_id)
    
    def _create_welcome_response(self, patient_id: str, start_time: datetime) -> Dict[str, Any]:
        """Create welcome response"""
        patient_name = self.current_patient_data.get('first_name', 'there')
        welcome_message = self._generate_welcome_message(patient_name)
        self.welcomed_patients.add(patient_id)
        
        self.memory.add_exchange("Session started", welcome_message, "welcome", ["Welcome Service"])
        
        return {
            "success": True,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "query": "Session started",
            "response": welcome_message,
            "response_format": "text",
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "agents_used": ["Welcome Service"],
            "is_welcome_message": True
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        if not self.memory.exchanges:
            return {
                "current_patient": {
                    "id": self.memory.current_patient_id,
                    "name": self.current_patient_data.get('first_name', 'Unknown')
                },
                "session_status": "No conversation yet",
                "conversation_length": 0
            }
        
        intents = [exchange.get('intent', 'unknown') for exchange in self.memory.exchanges]
        agents = []
        for exchange in self.memory.exchanges:
            agents.extend(exchange.get('agents_used', []))
        
        return {
            "current_patient": {
                "id": self.memory.current_patient_id,
                "name": self.current_patient_data.get('first_name', 'Unknown')
            },
            "session_duration": str(datetime.now() - self.memory.session_start),
            "conversation_length": len(self.memory.exchanges),
            "intent_distribution": {intent: intents.count(intent) for intent in set(intents)},
            "agents_used": list(set(agents)),
            "recent_exchanges": list(self.memory.exchanges)[-3:]
        }
    
    def clear_session(self):
        """Clear session"""
        try:
            patient_id = self.memory.current_patient_id
            self.memory.exchanges.clear()
            self.memory.patient_context = ""
            self.memory.session_start = datetime.now()
            self.welcomed_patients.clear()
            self.current_patient_data = {}
        except Exception as e:
            logger.error(f"Error clearing session: {e}")


# OPTIMIZED Terminal Interface - Minimal Output
async def optimized_terminal_interface():
    """Optimized terminal interface - minimal logging for Flask compatibility"""
    
    print("🏥 OPTIMIZED PATIENT AGENT - MINIMAL LOGGING")
    print("=" * 60)
    
    agent = EnhancedProfessionalPatientAgent()
    
    print("🔌 Connecting to database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database")
        return
    
    print("✅ System ready!")
    
    current_patient_id = None
    
    while True:
        try:
            if not current_patient_id:
                current_patient_id = input("\n🆔 Patient ID: ").strip()
                if not current_patient_id:
                    continue
            
            user_input = input(f"\n[{current_patient_id}] Question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower().startswith('switch '):
                new_patient_id = user_input[7:].strip()
                if new_patient_id:
                    current_patient_id = new_patient_id
                    print(f"🔄 Switched to: {current_patient_id}")
                continue
            elif user_input.lower() == 'clear':
                agent.clear_session()
                print("✅ Session cleared")
                continue
            
            print("Processing...")
            
            result = await agent.process_query(user_input, current_patient_id)
            
            # OPTIMIZED: Minimal output
            if result.get("response_format") == "json":
                print(f"\n📊 JSON Response: {result.get('facilities_found', 0)} facilities found")
                print(f"Location: {result.get('search_location', 'N/A')}")
                
                # Show first few facilities only
                if result.get('structured_data', {}).get('facilities'):
                    facilities = result['structured_data']['facilities'][:3]
                    print("Top facilities:")
                    for i, facility in enumerate(facilities, 1):
                        print(f"  {i}. {facility.get('name', 'Unknown')}")
            else:
                print(f"\n💬 Response:")
                print(result['response'])
            
            print(f"\n⏱️  {result.get('processing_time', 0):.1f}s | {result.get('intent_classified', 'N/A')} | {result.get('response_format', 'text')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # OPTIMIZED: Set logging to ERROR level only for production use
    logging.basicConfig(
        level=logging.ERROR,  # Only show errors
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(optimized_terminal_interface())