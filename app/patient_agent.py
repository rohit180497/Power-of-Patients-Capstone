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

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Standardized query intents"""
    CRISIS = "crisis"
    TBI_RELATED = "tbi_related" 
    MEDICAL_GENERAL = "medical_general"
    PATIENT_SELF_INQUIRY = "patient_self_inquiry"
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
    normalized_query: str
    intent: QueryIntent
    confidence: float
    urgency: UrgencyLevel
    is_medical: bool
    requires_multi_agent: bool
    context_needed: bool
    reasoning: str
    medical_entities: List[str]

@dataclass
class AgentResponse:
    """Standardized agent response structure"""
    agent_name: str
    response: str
    confidence: float
    source_quality: str
    processing_time: float
    error: Optional[str] = None

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
            "response": response,
            "intent": intent,
            "agents_used": agents_used
        })
    
    def get_context(self, max_exchanges: int = 3) -> str:
        """Get formatted conversation context"""
        if not self.exchanges:
            return "No previous conversation."
        
        recent = list(self.exchanges)[-max_exchanges:]
        context_parts = ["=== Recent Conversation ==="]
        
        for i, exchange in enumerate(recent, 1):
            context_parts.append(f"{i}. Patient: {exchange['query']}")
            context_parts.append(f"   Assistant: {exchange['response'][:200]}{'...' if len(exchange['response']) > 200 else ''}")
        
        return "\n".join(context_parts)

class IntelligentQueryClassifier:
    """Advanced LLM-based query classifier"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Medical entity patterns for enhanced classification
        self.tbi_indicators = [
            "traumatic brain injury", "tbi", "concussion", "head injury", "brain damage",
            "post-concussion syndrome", "mild traumatic brain injury", "mtbi", 
            "closed head injury", "brain trauma", "head trauma", "cognitive symptoms",
            "memory problems after injury", "confusion after head hit", "dizziness after accident"
        ]
    
    async def classify_query(self, query: str, conversation_context: str = "") -> QueryAnalysis:
        """Comprehensive query classification using LLM"""
        
        classification_prompt = f"""
You are an expert medical query classifier for a healthcare AI system. Analyze this patient query with high precision.

CONVERSATION CONTEXT:
{conversation_context}

PATIENT QUERY: "{query}"

CLASSIFICATION TASK:
Determine the query's intent, medical relevance, and routing requirements.

TBI-RELATED INDICATORS:
- Traumatic brain injury, TBI, concussion, head injury, brain trauma
- Post-concussion symptoms: headaches, dizziness, memory issues, confusion
- Cognitive problems after head injury
- Brain damage, closed head injury, mild TBI (mTBI)
- Sports-related head injuries, car accident brain injuries
- Recovery from head trauma, TBI rehabilitation

INTENT CATEGORIES:
1. "crisis" - Suicide ideation, severe mental health emergency, immediate danger
2. "tbi_related" - Any TBI, concussion, brain injury, or related symptoms/questions
3. "medical_general" - Other medical conditions, medications, treatments, health concerns
4. "patient_self_inquiry" - Patient asking about their own medical records/information
5. "continuation" - Follow-up responses (yes/no), clarifications to previous medical discussion
6. "general_conversation" - Greetings, thanks, non-medical conversation

URGENCY LEVELS:
- "emergency" - Immediate medical attention needed, crisis situations
- "high" - Serious symptoms, significant health concerns requiring prompt attention
- "medium" - Important health questions, symptom inquiries, medical guidance needed
- "low" - General information, follow-ups, non-urgent medical questions

ANALYSIS REQUIREMENTS:
1. Normalize the query (clear, explicit version)
2. Identify medical entities and TBI-related terms
3. Determine if multiple agents should process this query
4. Assess urgency and medical relevance
5. Consider conversation context for continuations

Respond in JSON format:
{{
    "normalized_query": "clear, explicit version addressing pronouns and context",
    "intent": "one of the 6 intents above",
    "confidence": 0.0-1.0,
    "urgency": "emergency/high/medium/low", 
    "is_medical": true/false,
    "requires_multi_agent": true/false,
    "context_needed": true/false,
    "reasoning": "detailed explanation of classification decisions",
    "medical_entities": ["list", "of", "medical", "terms", "found"],
    "tbi_related": true/false
}}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    classification_prompt,
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
            
            # Map string intent to enum
            intent_mapping = {
                "crisis": QueryIntent.CRISIS,
                "tbi_related": QueryIntent.TBI_RELATED,
                "medical_general": QueryIntent.MEDICAL_GENERAL,
                "patient_self_inquiry": QueryIntent.PATIENT_SELF_INQUIRY,
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
                normalized_query=result.get("normalized_query", query),
                intent=intent_mapping.get(result.get("intent"), QueryIntent.GENERAL_CONVERSATION),
                confidence=result.get("confidence", 0.5),
                urgency=urgency_mapping.get(result.get("urgency"), UrgencyLevel.LOW),
                is_medical=result.get("is_medical", False),
                requires_multi_agent=result.get("requires_multi_agent", False),
                context_needed=result.get("context_needed", False),
                reasoning=result.get("reasoning", ""),
                medical_entities=result.get("medical_entities", [])
            )
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> QueryAnalysis:
        """Fallback classification when LLM fails"""
        query_lower = query.lower()
        
        # Emergency detection
        if any(term in query_lower for term in ['suicide', 'kill myself', 'emergency', 'crisis']):
            intent = QueryIntent.CRISIS
            urgency = UrgencyLevel.EMERGENCY
        elif any(term in query_lower for term in self.tbi_indicators):
            intent = QueryIntent.TBI_RELATED
            urgency = UrgencyLevel.MEDIUM
        elif any(term in query_lower for term in ['my', 'mine', 'about me', 'my records']):
            intent = QueryIntent.PATIENT_SELF_INQUIRY
            urgency = UrgencyLevel.MEDIUM
        else:
            intent = QueryIntent.GENERAL_CONVERSATION
            urgency = UrgencyLevel.LOW
        
        return QueryAnalysis(
            original_query=query,
            normalized_query=query,
            intent=intent,
            confidence=0.3,
            urgency=urgency,
            is_medical=(intent in [QueryIntent.TBI_RELATED, QueryIntent.MEDICAL_GENERAL, QueryIntent.CRISIS]),
            requires_multi_agent=(intent == QueryIntent.TBI_RELATED),
            context_needed=(intent == QueryIntent.PATIENT_SELF_INQUIRY),
            reasoning="Fallback classification due to LLM error",
            medical_entities=[]
        )

class MultiAgentOrchestrator:
    """Orchestrates multiple agents for comprehensive responses"""
    
    def __init__(self, medical_agent, tbi_agent):
        self.medical_agent = medical_agent
        self.tbi_agent = tbi_agent
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    
    async def process_with_multiple_agents(self, query: str, analysis: QueryAnalysis) -> List[AgentResponse]:
        """Process query with multiple agents in parallel"""
        tasks = []
        
        # Always include medical agent for medical queries
        if analysis.is_medical and self.medical_agent:
            tasks.append(self._process_with_medical_agent(query))
        
        # Include TBI agent for TBI-related queries
        if analysis.intent == QueryIntent.TBI_RELATED and self.tbi_agent:
            tasks.append(self._process_with_tbi_agent(query))
        
        if not tasks:
            return []
        
        # Execute agents in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        valid_responses = []
        for response in responses:
            if not isinstance(response, Exception):
                valid_responses.append(response)
        
        return valid_responses
    
    async def _process_with_medical_agent(self, query: str) -> AgentResponse:
        """Process with medical agent"""
        start_time = datetime.now()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.medical_agent.process_query(query)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name="Medical Agent (MedPalm)",
                response=response,
                confidence=0.85,
                source_quality="high",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                agent_name="Medical Agent (MedPalm)",
                response="",
                confidence=0.0,
                source_quality="error",
                processing_time=processing_time,
                error=str(e)
            )
    
    async def _process_with_tbi_agent(self, query: str) -> AgentResponse:
        """Process with TBI agent"""
        start_time = datetime.now()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.tbi_agent.ask_question(query, top_k=8)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                agent_name="TBI Specialist",
                response=result['answer'],
                confidence=0.75,
                source_quality="medium",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                agent_name="TBI Specialist",
                response="",
                confidence=0.0,
                source_quality="error",
                processing_time=processing_time,
                error=str(e)
            )

class IntelligentResponseSynthesizer:
    """Synthesizes multiple agent responses into comprehensive answers"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def synthesize_response(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        agent_responses: List[AgentResponse],
        patient_context: str = "",
        conversation_context: str = ""
    ) -> str:
        """Synthesize comprehensive response from multiple agents"""
        
        # Handle crisis situations immediately
        if analysis.intent == QueryIntent.CRISIS:
            return self._get_crisis_response()
        
        # If no agent responses, provide fallback
        if not agent_responses:
            return self._get_fallback_response(query, analysis)
        
        # Filter successful responses
        valid_responses = [r for r in agent_responses if not r.error]
        if not valid_responses:
            return self._get_error_response()
        
        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            query, analysis, valid_responses, patient_context, conversation_context
        )
        
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
            # Return best available response
            best_response = max(valid_responses, key=lambda r: r.confidence)
            return self._add_medical_disclaimer(best_response.response) if analysis.is_medical else best_response.response
    
    def _build_synthesis_prompt(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        responses: List[AgentResponse],
        patient_context: str,
        conversation_context: str
    ) -> str:
        """Build comprehensive synthesis prompt"""
        
        # Prepare agent responses section
        agent_sections = []
        for i, response in enumerate(responses, 1):
            quality_indicator = "🔹" if response.source_quality == "high" else "🔸" if response.source_quality == "medium" else "🔺"
            agent_sections.append(f"""
{quality_indicator} **{response.agent_name}** (Confidence: {response.confidence:.2f})
Response: {response.response}
Processing Time: {response.processing_time:.2f}s
""")
        
        synthesis_prompt = f"""
You are Sallie, a professional and empathetic healthcare assistant created by Power of Patients. Synthesize the following agent responses into a comprehensive, helpful answer for the patient.

PATIENT QUERY: "{query}"
NORMALIZED QUERY: "{analysis.normalized_query}"
QUERY INTENT: {analysis.intent.value}
URGENCY LEVEL: {analysis.urgency.value}
MEDICAL ENTITIES: {', '.join(analysis.medical_entities) if analysis.medical_entities else 'None identified'}

{conversation_context}

PATIENT MEDICAL CONTEXT:
{patient_context if patient_context else 'Patient context not available for this response.'}

SPECIALIST AGENT RESPONSES:
{''.join(agent_sections)}

SYNTHESIS GUIDELINES:
1. **Comprehensive Coverage**: Combine insights from all agents to provide complete information
2. **Patient-Centric**: Write for patients, not medical professionals - use clear, understandable language
3. **Empathetic Tone**: Be supportive and understanding, especially for health concerns
4. **Structured Information**: Organize information logically (symptoms, causes, treatments, next steps)
5. **Actionable Guidance**: Provide clear next steps and recommendations
6. **Source Integration**: Seamlessly blend information without saying "Agent X said..."
7. **Completeness**: Address all aspects of the patient's question thoroughly

SPECIFIC INSTRUCTIONS FOR TBI QUERIES:
- Provide detailed information about symptoms, causes, and management
- Include both immediate and long-term considerations
- Mention different types of TBI (mild, moderate, severe)
- Discuss recovery timelines and rehabilitation
- Address common concerns patients have

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Detailed explanation with examples
3. What to expect/look for
4. When to seek medical care
5. Supportive closing with offer to help further

Generate a comprehensive, patient-friendly response that synthesizes ALL available information:"""

        return synthesis_prompt
    
    def _get_crisis_response(self) -> str:
        """Crisis response for mental health emergencies"""
        return """
🚨 **I'm very concerned about what you're sharing with me. Your life has value and there are people who want to help you right now.**

**IMMEDIATE HELP AVAILABLE 24/7:**
• **National Suicide Prevention Lifeline: 988** (US)
• **Crisis Text Line: Text HOME to 741741**
• **International: befrienders.org**

**Please reach out immediately to:**
• Emergency services (911) if in immediate danger
• Your healthcare provider or psychiatrist
• A trusted friend or family member
• Local hospital emergency department
• Local crisis intervention center

**You are not alone.** Professional counselors are standing by 24/7 to provide support and help you work through these feelings safely.

Is there someone you can call right now? I can help you find local mental health resources if you'd like.

**Remember: This crisis will pass, and there are effective treatments and support available.**
"""
    
    def _get_fallback_response(self, query: str, analysis: QueryAnalysis) -> str:
        """Fallback response when no agents are available"""
        if analysis.intent == QueryIntent.TBI_RELATED:
            return f"""I understand you're asking about {analysis.normalized_query}. This is an important health concern that deserves comprehensive information.

While I'm experiencing some technical difficulties accessing my medical databases right now, I strongly recommend:

**Immediate Steps:**
• Consult with your healthcare provider or neurologist
• If this is about recent head trauma, consider emergency care
• Keep track of any symptoms you're experiencing

**For TBI-related concerns:**
• Contact a neurologist or brain injury specialist
• Reach out to brain injury support organizations
• Consider neuropsychological evaluation if symptoms persist

I apologize that I can't provide the detailed information you deserve right now. Your health concerns are important, and professional medical guidance would be most appropriate for TBI-related questions."""

        return f"I understand you're asking about {analysis.normalized_query}. While I'm having some technical difficulties right now, I recommend consulting with your healthcare provider for proper guidance on this matter."
    
    def _get_error_response(self) -> str:
        """Response when all agents fail"""
        return """I apologize, but I'm experiencing technical difficulties accessing my medical information systems right now. 

For your health concerns, I recommend:
• Consulting directly with your healthcare provider
• Contacting your doctor's office
• Using telehealth services if available
• Visiting urgent care for non-emergency concerns

Your health questions are important and deserve proper attention from medical professionals."""
    
    def _add_medical_disclaimer(self, response: str) -> str:
        """Add appropriate medical disclaimer"""
        disclaimer = """

⚠️ **Medical Disclaimer:** I am an AI assistant providing general health information for educational purposes only. This information should not replace professional medical consultation. Please consult with your healthcare provider, doctor, or qualified medical professional for personalized medical advice, proper diagnosis, and treatment options specific to your condition."""
        
        return response + disclaimer

class EnhancedProfessionalPatientAgent:
    """Enhanced Professional Patient Agent with intelligent multi-agent architecture"""
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None):
        """Initialize the Enhanced Patient Agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        # Database connection
        self.db_connection = None
        
        # Enhanced conversation memory
        self.memory = ConversationMemory(
            exchanges=deque(maxlen=20),
            current_patient_id="",
            patient_context=""
        )
        
        # Patient data
        self.current_patient_data = {}
        self.welcomed_patients = set()
        
        # Initialize core components
        self._initialize_core_components()
        self._initialize_specialized_agents()
        
        logger.info("✅ Enhanced Professional Patient Agent initialized with multi-agent architecture")
    
    def _initialize_core_components(self):
        """Initialize core LLM components"""
        try:
            if not self.gemini_api_key:
                raise ValueError("Gemini API key required")
            
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize intelligent components
            self.query_classifier = IntelligentQueryClassifier(self.gemini_model)
            self.response_synthesizer = IntelligentResponseSynthesizer(self.gemini_model)
            
            logger.info("✅ Core LLM components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise
    
    def _initialize_specialized_agents(self):
        """Initialize and validate specialized agents"""
        # Initialize Medical Agent
        try:
            self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
            logger.info("✅ Medical Agent (MedPalm) initialized")
        except Exception as e:
            logger.warning(f"⚠️ Medical Agent not available: {e}")
            self.medical_agent = None
        
        # Initialize TBI Agent
        try:
            self.tbi_agent = CDCTBIRetriever(
                pinecone_api_key=self.pinecone_api_key,
                index_name=os.getenv("PINECONE_INDEX2_NAME"),
                embedding_model=os.getenv("EMBEDDING_MODEL"),
                llm_provider="gemini"
            )
            logger.info("✅ TBI Specialist Agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ TBI Specialist Agent not available: {e}")
            self.tbi_agent = None
        
        # Initialize Multi-Agent Orchestrator
        self.orchestrator = MultiAgentOrchestrator(self.medical_agent, self.tbi_agent)
        logger.info("✅ Multi-Agent Orchestrator initialized")
    
    async def connect_to_database(self, db_config: Dict[str, str] = None) -> bool:
        """Connect to database with enhanced error handling"""
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
            
            logger.info("✅ Database connected successfully")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Database connection failed: {e}")
            return False
    
    async def load_patient_data(self, patient_id: str) -> bool:
        """Load comprehensive patient data with better error handling"""
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
                logger.error(f"❌ Patient not found: {patient_id}")
                return False
            
            self.current_patient_data = patient_data
            self.memory.current_patient_id = patient_id
            self.memory.patient_context = self._build_comprehensive_patient_context(patient_data)
            
            logger.info(f"✅ Patient data loaded: {patient_id}")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Error loading patient data: {e}")
            return False
    
    def _check_db_connection(self) -> bool:
        """Check database connection health"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
        except:
            return False
        return False
    
    def _build_comprehensive_patient_context(self, patient_data: Dict) -> str:
        """Build detailed patient context for better AI responses"""
        context_parts = [
            "=== PATIENT PROFILE ===",
            f"Patient ID: {patient_data.get('patient_id', 'N/A')}",
            f"Name: {patient_data.get('first_name', 'N/A')}",
            f"Age: {patient_data.get('age', 'N/A')}",
            f"Location: {patient_data.get('city', 'N/A')}, {patient_data.get('country', 'N/A')}",
            f"Patient Type: {patient_data.get('patient_type', 'N/A')}",
            f"Registration Date: {patient_data.get('registered_at', 'N/A')}",
            "",
            "=== TBI/INJURY HISTORY ===",
            f"Has Previous TBI: {patient_data.get('has_tbi_before', 'N/A')}",
            f"Total TBI Incidents: {patient_data.get('total_tbi', 'N/A')}",
            f"Most Recent Incident: {patient_data.get('tbi_incident_date', 'N/A')}",
            f"Injury Source: {patient_data.get('injury_from', 'N/A')}",
            f"Head Impact Location: {patient_data.get('head_hit_location', 'N/A')}",
            f"Event Description: {patient_data.get('describe_event', 'N/A')}",
            "",
            "=== SYMPTOM PROFILE ===",
            f"Immediate Post-Injury Symptoms: {patient_data.get('immediate_symptoms_resulting', 'N/A')}",
            f"Most Severe Symptoms: {patient_data.get('worst_symptoms', 'N/A')}"
        ]
        
        # Process JSON fields with better error handling
        json_fields = {
            'symptom_json': 'Current Symptoms',
            'sdoh_json': 'Social Determinants of Health',
            'therapy_json': 'Therapy/Treatment History'
        }
        
        for field, title in json_fields.items():
            if patient_data.get(field):
                try:
                    json_data = json.loads(patient_data[field]) if isinstance(patient_data[field], str) else patient_data[field]
                    context_parts.append(f"\n=== {title.upper()} ===")
                    if isinstance(json_data, dict):
                        for key, value in json_data.items():
                            context_parts.append(f"{key}: {value}")
                    else:
                        context_parts.append(str(json_data))
                except Exception as e:
                    context_parts.append(f"\n=== {title.upper()} ===")
                    context_parts.append(f"Raw data: {patient_data[field]}")
        
        return "\n".join(context_parts)
    
    def _generate_personalized_welcome(self, patient_name: str) -> str:
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

I'm Sallie, your professional healthcare assistant created by Power of Patients. I'm here to provide you with comprehensive, personalized support for your medical questions and health concerns.

I have access to your medical profile and can help you with:
• **TBI and concussion information** - symptoms, recovery, management
• **General medical guidance** - conditions, treatments, medications
• **Your personal health questions** - based on your medical history
• **Healthcare navigation** - finding resources and next steps

How can I help you with your health concerns today?"""
    
    async def process_query(self, query: str, patient_id: str) -> Dict[str, Any]:
        """
        Enhanced query processing with intelligent multi-agent architecture
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Ensure patient data is loaded
            if (not self.current_patient_data or 
                self.current_patient_data.get('patient_id') != patient_id):
                
                await self._handle_patient_session_setup(patient_id)
                
                if not self.current_patient_data:
                    return self._create_error_response(
                        f"Could not load patient data for {patient_id}",
                        "I'm sorry, I couldn't access your medical records. Please verify your patient ID.",
                        start_time
                    )
                
                # Send welcome for new sessions
                if patient_id not in self.welcomed_patients:
                    return self._create_welcome_response(patient_id, start_time)
            
            # Step 2: Intelligent Query Classification
            logger.info(f"🧠 Classifying query: {query[:50]}...")
            conversation_context = self.memory.get_context(3)
            analysis = await self.query_classifier.classify_query(query, conversation_context)
            
            logger.info(f"🎯 Classified as: {analysis.intent.value} (confidence: {analysis.confidence:.2f})")
            
            # Step 3: Multi-Agent Processing
            agent_responses = []
            if analysis.is_medical:
                logger.info(f"🚀 Processing with multiple agents...")
                agent_responses = await self.orchestrator.process_with_multiple_agents(query, analysis)
                logger.info(f"📊 Received {len(agent_responses)} agent responses")
            
            # Step 4: Intelligent Response Synthesis
            logger.info(f"🔄 Synthesizing comprehensive response...")
            patient_context = self.memory.patient_context if analysis.context_needed else ""
            
            final_response = await self.response_synthesizer.synthesize_response(
                query, analysis, agent_responses, patient_context, conversation_context
            )
            
            # Step 5: Update Memory
            agents_used = [r.agent_name for r in agent_responses] if agent_responses else ["General Assistant"]
            self.memory.add_exchange(query, final_response, analysis.intent.value, agents_used)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Query processed successfully in {processing_time:.2f}s")
            
            return {
                "success": True,
                "patient_id": patient_id,
                "patient_name": self.current_patient_data.get('first_name', 'Unknown'),
                "query": query,
                "normalized_query": analysis.normalized_query if analysis.normalized_query != query else None,
                "intent_classified": analysis.intent.value,
                "urgency_level": analysis.urgency.value,
                "agents_used": agents_used,
                "response": final_response,
                "processing_time": processing_time,
                "confidence": analysis.confidence,
                "medical_entities": analysis.medical_entities,
                "multi_agent_used": len(agent_responses) > 1
            }
            
        except Exception as e:
            logger.exception(f"❌ Error processing query: {e}")
            return self._create_error_response(str(e), "I encountered an error processing your request. Please try again.", start_time)
    
    async def _handle_patient_session_setup(self, patient_id: str):
        """Handle patient session setup and data loading"""
        logger.info(f"🔄 Setting up session for patient: {patient_id}")
        
        # Reset memory for new patient
        if self.memory.current_patient_id != patient_id:
            self.memory.exchanges.clear()
            self.welcomed_patients.discard(patient_id)
        
        # Load patient data
        await self.load_patient_data(patient_id)
    
    def _create_welcome_response(self, patient_id: str, start_time: datetime) -> Dict[str, Any]:
        """Create welcome response for new sessions"""
        patient_name = self.current_patient_data.get('first_name', 'there')
        welcome_message = self._generate_personalized_welcome(patient_name)
        self.welcomed_patients.add(patient_id)
        
        # Add to memory
        self.memory.add_exchange("Session started", welcome_message, "welcome", ["Welcome Service"])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "query": "Session started",
            "intent_classified": "welcome_message",
            "agents_used": ["Welcome Service"],
            "response": welcome_message,
            "processing_time": processing_time,
            "is_welcome_message": True
        }
    
    def _create_error_response(self, error: str, response: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error response"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": False,
            "error": error,
            "response": response,
            "processing_time": processing_time
        }
    
    def get_comprehensive_session_summary(self) -> Dict[str, Any]:
        """Get detailed session summary with analytics"""
        if not self.memory.exchanges:
            return {
                "current_patient": {
                    "id": self.memory.current_patient_id,
                    "name": self.current_patient_data.get('first_name', 'Unknown')
                },
                "session_status": "No conversation yet",
                "conversation_length": 0
            }
        
        # Analyze conversation patterns
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
            "recent_exchanges": list(self.memory.exchanges)[-3:],
            "memory_usage": f"{len(self.memory.exchanges)}/20"
        }
    
    def clear_session(self):
        """Clear session with proper cleanup"""
        try:
            patient_id = self.memory.current_patient_id
            
            # Clear memory
            self.memory.exchanges.clear()
            self.memory.patient_context = ""
            self.memory.session_start = datetime.now()
            
            # Reset welcome status
            self.welcomed_patients.clear()
            
            # Clear patient data
            self.current_patient_data = {}
            
            logger.info(f"✅ Session cleared completely (Patient: {patient_id})")
            
        except Exception as e:
            logger.error(f"❌ Error clearing session: {e}")


# Enhanced Terminal Interface for Testing
async def enhanced_terminal_interface():
    """Enhanced terminal interface with better UX"""
    print("=" * 100)
    print("🏥 ENHANCED PROFESSIONAL PATIENT AGENT - INTELLIGENT MULTI-AGENT SYSTEM")
    print("=" * 100)
    
    # Initialize agent
    print("🚀 Initializing Enhanced Patient Agent...")
    agent = EnhancedProfessionalPatientAgent()
    
    # Connect to database
    print("🔌 Connecting to database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database")
        return
    
    print("✅ Enhanced Patient Agent initialized successfully!")
    print("\n🎯 **KEY FEATURES:**")
    print("   • **Intelligent Query Classification** - Advanced LLM-based intent recognition")
    print("   • **Multi-Agent Processing** - TBI queries use both TBI + Medical agents in parallel")
    print("   • **Response Synthesis** - LLM combines multiple agent responses comprehensively")
    print("   • **Patient-Centric Design** - Responses optimized for patient understanding")
    print("   • **Enhanced Context Awareness** - Sophisticated conversation memory")
    
    print("\n📋 **COMMANDS:**")
    print("   • Just type your health questions naturally!")
    print("   • 'switch <patient_id>' - Change patient")
    print("   • 'summary' - View detailed session analytics")
    print("   • 'test tbi' - Test TBI multi-agent processing")
    print("   • 'clear' - Clear session")
    print("   • 'quit' - Exit")
    print("=" * 100)
    
    current_patient_id = None
    
    while True:
        try:
            # Get patient ID if not set
            if not current_patient_id:
                current_patient_id = input("\n🆔 Enter Patient ID: ").strip()
                if not current_patient_id:
                    continue
            
            # Get user input
            user_input = input(f"\n[Patient {current_patient_id}] 💬 Your Question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("\n👋 Thank you for using the Enhanced Patient Agent!")
                break
            elif user_input.lower().startswith('switch '):
                new_patient_id = user_input[7:].strip()
                if new_patient_id:
                    current_patient_id = new_patient_id
                    print(f"🔄 Switched to patient: {current_patient_id}")
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_comprehensive_session_summary()
                print("\n📊 **SESSION ANALYTICS:**")
                print(json.dumps(summary, indent=2, default=str))
                continue
            elif user_input.lower() == 'test tbi':
                user_input = "What are the major symptoms of traumatic brain injury and how should I manage them?"
                print(f"🧪 Testing TBI multi-agent processing with: {user_input}")
            elif user_input.lower() == 'clear':
                agent.clear_session()
                print("✅ Session cleared")
                continue
            
            # Process query with timing
            print(f"🤖 Processing query with intelligent multi-agent system...")
            start_time = datetime.now()
            
            result = await agent.process_query(user_input, current_patient_id)
            
            if result["success"]:
                print(f"\n💬 **Sallie:** {result['response']}")
                
                # Show detailed analytics
                print(f"\n📊 **PROCESSING ANALYTICS:**")
                print(f"   🎯 Intent: {result['intent_classified']}")
                print(f"   🚀 Agents Used: {', '.join(result['agents_used'])}")
                print(f"   ⚡ Processing Time: {result['processing_time']:.2f}s")
                print(f"   🎚️ Confidence: {result['confidence']:.2f}")
                print(f"   🚨 Urgency: {result['urgency_level']}")
                
                if result.get('normalized_query'):
                    print(f"   🔄 Normalized: '{result['normalized_query']}'")
                if result.get('medical_entities'):
                    print(f"   🏥 Medical Entities: {', '.join(result['medical_entities'])}")
                if result.get('multi_agent_used'):
                    print(f"   🤝 Multi-Agent Synthesis: ✅")
                
            else:
                print(f"\n❌ **Error:** {result['error']}")
                print(f"**Response:** {result['response']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(enhanced_terminal_interface())