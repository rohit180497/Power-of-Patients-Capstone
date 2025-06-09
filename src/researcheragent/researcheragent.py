"""
Enhanced Professional Researcher Agent with LLM-powered intelligence
Industry-standard implementation with intelligent routing, memory management, and fallback systems
"""

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

# Import specialized agents
from src.pandasagent.pandasagent2 import EnhancedPandasAgent, DB_SCHEMA

try:
    from src.medpalm.medical_assistant import MedPalmAgent
except ImportError:
    MedPalmAgent = None
    
try:
    from src.retrieval.cdcretrieval import CDCTBIRetriever
except ImportError:
    CDCTBIRetriever = None

logger = logging.getLogger(__name__)

@dataclass
class ResearchQueryAnalysis:
    """Structured analysis of researcher query with LLM insights"""
    original_query: str
    paraphrased_query: str
    intent: str
    confidence: float
    is_continuation: bool
    is_data_query: bool
    requires_visualization: bool
    urgency_level: str
    reasoning: str
    suggested_agent: str

@dataclass
class ResearchMemory:
    """Enhanced conversation memory for research context"""
    exchanges: deque
    current_researcher_id: str
    last_agent_used: str
    pending_followup: Optional[str]
    analysis_history: List[str]
    
    def add_exchange(self, query: str, response: str, agent_used: str, metadata: Dict = None):
        self.exchanges.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "agent": agent_used,
            "metadata": metadata or {}
        })
        self.last_agent_used = agent_used
        
        # Track analysis types
        if metadata and metadata.get('intent') == 'data_analysis':
            self.analysis_history.append(query)
    
    def get_recent_context(self, num_exchanges: int = 5) -> str:
        """Get formatted recent conversation context with FULL responses for data continuity"""
        if not self.exchanges:
            return ""
        
        recent = list(self.exchanges)[-num_exchanges:]
        context_parts = []
        for exchange in recent:
            context_parts.append(f"Researcher: {exchange['query']}")
            # Keep FULL responses for continuation queries to preserve data context
            response = exchange['response']
            # Only truncate if response is extremely long (>2000 chars) to preserve data tables
            if len(response) > 2000:
                # For data tables, try to preserve the table structure
                if '<table' in response or '|' in response:
                    # Keep first 1500 chars which should include most tables
                    response = response[:1500] + "...\n[Data table continues...]"
                else:
                    response = response[:500] + "..."
            context_parts.append(f"Assistant: {response}")
        
        return "\n".join(context_parts)

class LLMResearchGuardrailAgent:
    """Intelligent LLM-powered guardrail system for research queries"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def analyze_research_query_safety(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Analyze research query safety with academic context"""
        
        json_template = """{
    "allow": true/false,
    "confidence": 0.0-1.0,
    "category": "research_query|data_analysis|medical_research|academic_discussion|inappropriate",
    "redirect_message": "message if blocked",
    "reasoning": "detailed explanation"
}"""

        guardrail_prompt = f"""
You are a research AI guardrail system for academic and medical research contexts.

CONVERSATION CONTEXT:
{conversation_context}

RESEARCH QUERY: "{query}"

RESEARCH GUIDELINES:
✅ ALLOW - All legitimate research queries, data analysis, medical research, academic discussions
✅ ALLOW - Database queries, statistical analysis, visualization requests
✅ ALLOW - Medical information for research purposes
✅ ALLOW - TBI research, methodology questions, platform information
✅ ALLOW - Continuation responses and follow-up questions
✅ ALLOW - Identity questions about Sallie or Power of Patients
✅ ALLOW - Patient demographic information for research purposes (this is research data)
✅ ALLOW - Specific patient information requests for academic research
✅ ALLOW - Patient details, demographics, medical information when used for research analysis

❌ BLOCK ONLY - Clearly harmful, illegal, or completely non-research related content
❌ BLOCK ONLY - Personal attacks, harassment, or inappropriate behavior
❌ BLOCK ONLY - Requests that could harm patients if this were real clinical data (but this is research data)

IMPORTANT RESEARCH CONTEXT:
- This is a RESEARCH environment with anonymized research data
- Researchers need access to patient demographics for legitimate research analysis
- Patient information requests are part of standard research methodology
- Demographics, symptom data, and patient details are expected research queries

ANALYSIS REQUIRED:
1. Is this a legitimate research or academic query?
2. Is this related to medical/health research?
3. Is this about data analysis or methodology?
4. Could this query advance research understanding?
5. Is this continuation of research discussion?
6. Is this requesting patient data for research purposes?

NOTE: Be VERY permissive for research contexts. Patient demographic requests are normal in research. Only block clearly inappropriate content.

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
                    "allow": result.get("allow", True),  # Default allow for research
                    "confidence": result.get("confidence", 0.5),
                    "category": result.get("category", "research_query"),
                    "redirect_message": result.get("redirect_message", ""),
                    "reasoning": result.get("reasoning", "")
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse research guardrail JSON: {result_text}")
                return {"allow": True, "confidence": 0.3, "category": "parse_error", "redirect_message": "", "reasoning": "Failed to parse response"}
        
        except Exception as e:
            logger.error(f"Research guardrail analysis failed: {e}")
            return {"allow": True, "confidence": 0.1, "category": "error", "redirect_message": "", "reasoning": "Guardrail system error"}

class LLMResearchQueryAnalyzer:
    """Intelligent research query analysis using LLM with conversation memory"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def analyze_research_query(self, query: str, memory: ResearchMemory) -> ResearchQueryAnalysis:
        """Comprehensive research query analysis with memory context including FULL data from previous responses"""
        
        # Get conversation context with FULL recent responses for data continuity
        conversation_context = memory.get_recent_context(3)  # Include full context for better understanding
        researcher_id = memory.current_researcher_id
        
        json_template = """{
    "paraphrased_query": "clear, explicit version of the query",
    "intent": "primary intent from options below",
    "confidence": 0.0-1.0,
    "is_continuation": true/false,
    "is_data_query": true/false,
    "requires_visualization": true/false,
    "urgency_level": "low/medium/high",
    "reasoning": "detailed explanation of analysis decisions",
    "suggested_agent": "pandas|medical|tbi|schema|methodology|general"
}"""

        analysis_prompt = f"""
You are an advanced research query analyzer. Analyze this query in context of ongoing conversation with FULL ACCESS to previous data.

RESEARCHER ID: {researcher_id}
CONVERSATION CONTEXT (includes full data from previous responses):
{conversation_context}

ORIGINAL QUERY: "{query}"

ANALYSIS TASKS:
1. **QUERY PARAPHRASING**: Rewrite query to be explicit, resolving pronouns and references using COMPLETE conversation context including specific data values from previous responses
2. **INTENT CLASSIFICATION**: Determine primary research intent
3. **AGENT ROUTING**: Decide which specialist should handle this
4. **CONTEXT ANALYSIS**: Assess continuation and data requirements

CRITICAL PARAPHRASING RULES:
- If query refers to "that patient", "those results", "this data" - extract SPECIFIC values from conversation context
- If previous response contained a data table with patient IDs, use the ACTUAL patient ID when paraphrasing
- If query asks about "the top patient" or "highest", reference the specific patient ID from the table
- Include actual numbers, percentages, and identifiers from previous responses in paraphrased query
- For continuation queries, be EXTREMELY specific about what data the researcher is referring to

CRITICAL ROUTING RULES:
- If query asks for "details", "demographics", "information", "profile" of a specific patient → data_analysis + pandas (NOT continuation)
- If query asks for "more data", "patient info", "background" of a specific patient → data_analysis + pandas  
- If query is just "yes/no/maybe/thanks" responses → continuation + general
- If query asks for analysis of specific patient data → data_analysis + pandas

EXAMPLES OF GOOD PARAPHRASING:
- "who is that patient?" → "Who is patient ID 1ba40009-06c2-4ef6-912d-89123343015f who had 9354 symptom logs (21.14% of total) as shown in the previous analysis?"
- "i want details of that patient" → "I want detailed information from the database about patient ID 5adc5975-8add-47ae-a796-6597d52c7145 who had 4470 symptom logs in 2022"
- "demographics of that patient" → "Show me demographic information for patient ID 5adc5975-8add-47ae-a796-6597d52c7145 from the patients table"

INTENT OPTIONS:
- "data_analysis" - Statistical analysis, data exploration, visualization requests requiring database queries OR requests for specific patient details/demographics
- "medical_general" - General medical questions not requiring database analysis  
- "tbi_knowledge" - TBI/concussion knowledge questions not requiring database queries
- "schema_info" - Database structure, tables, columns, data organization questions
- "methodology" - Research methods, statistical approaches, study design guidance
- "continuation" - Simple yes/no responses, thanks, or clarification requests (NOT patient detail requests)
- "platform_info" - Questions about Power of Patients, Sallie, or platform capabilities
- "general_conversation" - Greetings, thanks, casual interaction

AGENT ROUTING:
- "pandas" - For data analysis requiring database queries, visualizations, AND specific patient information requests
- "medical" - For general medical information and knowledge
- "tbi" - For TBI-specific knowledge and information
- "schema" - For database structure and organization questions
- "methodology" - For research methodology and statistical guidance
- "general" - For platform info, greetings, and general conversation

IMPORTANT: 
- Requests for patient details/demographics/information should be classified as "data_analysis" with "pandas" agent
- Only classify as "continuation" if it's a simple response like "yes", "no", "thanks"
- For continuation queries referring to previous data, make the paraphrased query EXTREMELY specific with actual patient IDs, numbers, and values

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
                
                return ResearchQueryAnalysis(
                    original_query=query,
                    paraphrased_query=result.get("paraphrased_query", query),
                    intent=result.get("intent", "general_conversation"),
                    confidence=result.get("confidence", 0.5),
                    is_continuation=result.get("is_continuation", False),
                    is_data_query=result.get("is_data_query", False),
                    requires_visualization=result.get("requires_visualization", False),
                    urgency_level=result.get("urgency_level", "low"),
                    reasoning=result.get("reasoning", "No reasoning provided"),
                    suggested_agent=result.get("suggested_agent", "general")
                )
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse research analysis JSON: {result_text}")
                return self._fallback_analysis(query)
        
        except Exception as e:
            logger.error(f"Research query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> ResearchQueryAnalysis:
        """Simple keyword-based fallback analysis when LLM fails (emergency only)"""
        
        logger.warning("Using emergency keyword-based fallback analysis - LLM analysis failed")
        
        query_lower = query.lower()
        
        # Patient detail detection
        if any(word in query_lower for word in ['details', 'demographics', 'information about', 'profile']):
            intent = "data_analysis"
            agent = "pandas"
        # Data analysis detection
        elif any(word in query_lower for word in ['show', 'analyze', 'chart', 'graph', 'distribution', 'correlation']):
            intent = "data_analysis"
            agent = "pandas"
        elif any(word in query_lower for word in ['table', 'schema', 'column', 'database']):
            intent = "schema_info"
            agent = "schema"
        elif any(word in query_lower for word in ['tbi', 'concussion', 'brain injury']):
            intent = "tbi_knowledge"
            agent = "tbi"
        elif any(word in query_lower for word in ['power of patients', 'sallie', 'who are you']):
            intent = "platform_info"
            agent = "general"
        else:
            intent = "general_conversation"
            agent = "general"
        
        return ResearchQueryAnalysis(
            original_query=query,
            paraphrased_query=query,
            intent=intent,
            confidence=0.3,
            is_continuation=False,
            is_data_query=(intent == "data_analysis"),
            requires_visualization=(intent == "data_analysis"),
            urgency_level="medium",
            reasoning="Emergency keyword-based fallback analysis due to LLM error",
            suggested_agent=agent
        )

class ProfessionalResearcherAgent:
    """
    Enhanced Professional Researcher Agent with LLM-powered intelligence
    """
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None):
        """Initialize the Enhanced Researcher Agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        # Database configuration
        self.db_config = {
            'user': os.getenv("user") or os.getenv("DB_USER"),
            'password': os.getenv("password") or os.getenv("DB_PASSWORD"),
            'host': os.getenv("host") or os.getenv("DB_HOST"),
            'port': os.getenv("port") or os.getenv("DB_PORT", "5432"),
            'dbname': os.getenv("dbname") or os.getenv("DB_NAME")
        }
        
        # Enhanced Memory Management
        self.memory = ResearchMemory(
            exchanges=deque(maxlen=30),  # Larger buffer for research context
            current_researcher_id="",
            last_agent_used="",
            pending_followup=None,
            analysis_history=[]
        )
        
        # Welcome tracking
        self.welcomed_researchers = set()
        
        # Company information
        self.company_info = """
POWER OF PATIENTS®

Power of Patients is a healthcare technology company designed to empower Traumatic Brain Injury (TBI) persons and their caregivers.

Mission: We want to help you tell your story, track your symptoms, provide education about clinical trials, and share treatment options.

Key Features:
• Patient empowerment platform for TBI survivors
• Symptom tracking and management tools
• Clinical trial information and education
• Treatment options and resources
• Caregiver support tools
• Comprehensive TBI database for research

Website: https://www.powerofpatients.com/

Sallie (me) is the AI healthcare assistant created by Power of Patients to help researchers, patients, and caregivers with TBI-related information, data analysis, and medical guidance.
"""
        
        # Initialize LLM and components
        self._initialize_llm_components()
        self._initialize_specialized_agents()
        
        logger.info("Enhanced Professional Researcher Agent initialized")
    
    def _initialize_llm_components(self):
        """Initialize LLM-powered components"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Initialize LLM-powered components
                self.query_analyzer = LLMResearchQueryAnalyzer(self.gemini_model)
                self.guardrail_agent = LLMResearchGuardrailAgent(self.gemini_model)
                
                logger.info("Research LLM components initialized successfully")
            else:
                logger.error("Gemini API key not provided")
                raise ValueError("Gemini API key required")
                
        except Exception as e:
            logger.error(f"Failed to initialize research LLM components: {e}")
            raise
    
    def _initialize_specialized_agents(self):
        """Initialize specialized research agents"""
        try:
            # Initialize PandasAgent for data analysis
            self.pandas_agent = EnhancedPandasAgent(gemini_api_key=self.gemini_api_key)
            logger.info("Enhanced PandasAgent initialized")
        except Exception as e:
            logger.error(f"PandasAgent initialization failed: {e}")
            self.pandas_agent = None
        
        try:
            # Initialize MedPalm Agent
            if MedPalmAgent and self.gemini_api_key:
                self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
                logger.info("MedPalm Agent initialized")
            else:
                self.medical_agent = None
                logger.warning("MedPalm Agent not available")
        except Exception as e:
            logger.warning(f"MedPalm Agent initialization failed: {e}")
            self.medical_agent = None
        
        try:
            # Initialize TBI Retrieval Agent
            if CDCTBIRetriever and self.pinecone_api_key:
                self.retrieval_agent = CDCTBIRetriever(
                    pinecone_api_key=self.pinecone_api_key,
                    index_name=os.getenv("PINECONE_INDEX2_NAME"),
                    embedding_model=os.getenv("EMBEDDING_MODEL"),
                    llm_provider="gemini"
                )
                logger.info("TBI Retrieval Agent initialized")
            else:
                self.retrieval_agent = None
                logger.warning("TBI Retrieval Agent not available")
        except Exception as e:
            logger.warning(f"TBI Retrieval Agent initialization failed: {e}")
            self.retrieval_agent = None
    
    async def connect_to_database(self) -> bool:
        """Connect PandasAgent to database with improved error handling"""
        try:
            if not self.pandas_agent:
                logger.error("PandasAgent not initialized")
                return False
            
            # Check required config
            required_keys = ['user', 'password', 'host', 'port', 'dbname']
            missing_keys = [key for key in required_keys if not self.db_config.get(key)]
            
            if missing_keys:
                logger.error(f"Missing database configuration: {missing_keys}")
                return False
            
            # Connect the PandasAgent to database
            success = self.pandas_agent.connect_to_database(self.db_config)
            
            if success:
                logger.info("Successfully connected PandasAgent to database")
                # Get schema info for context
                try:
                    self.schema_info = self.pandas_agent.get_comprehensive_schema_info()
                except:
                    self.schema_info = DB_SCHEMA
            else:
                logger.error("Failed to connect PandasAgent to database")
            
            return success
            
        except Exception as e:
            logger.exception(f"Database connection error: {e}")
            return False
    
    def _generate_researcher_welcome(self, researcher_id: str = "Researcher") -> str:
        """Generate personalized welcome message for researchers"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            greeting = "Good Morning"
        elif 12 <= current_hour < 17:
            greeting = "Good Afternoon" 
        elif 17 <= current_hour < 22:
            greeting = "Good Evening"
        else:
            greeting = "Hello"
        
        # Get database statistics if available
        db_stats = ""
        if hasattr(self, 'schema_info') and self.schema_info:
            lines = self.schema_info.split('\n')
            for line in lines:
                if "patients" in line and "Rows:" in line:
                    try:
                        patient_count = line.split('Rows:')[1].strip().split()[0]
                        db_stats = f"\n\nOur database currently contains {patient_count} patient records with comprehensive TBI incident data, symptoms, therapies, and social determinants."
                    except:
                        pass
                    break
        
        welcome_message = f"""{greeting}, {researcher_id}!

I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. I'm here to help you with TBI data analysis, medical questions, TBI information, and research guidance.{db_stats}

I can assist with:
• **Data Analysis**: Statistical analysis, demographics, treatment outcomes, visualizations
• **Medical Information**: General medical questions, conditions, treatments
• **TBI Knowledge**: Specific information about traumatic brain injury, concussions, symptoms
• **Research Methodology**: Statistical approaches, study design guidance

How can I assist you with your research analysis today?"""
        
        return welcome_message
    
    async def _handle_data_analysis(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle data analysis queries with comprehensive error handling and FOCUSED query processing"""
        try:
            if not self.pandas_agent:
                return {
                    "response": "Data analysis agent is not available. Please check system configuration.",
                    "visualization_html": None,
                    "has_visualization": False,
                    "agent_used": "PandasAgent (Unavailable)",
                    "metadata": {"error": "PandasAgent not initialized"}
                }
            
            # CRITICAL: Clean the query for PandasAgent - remove conversational context, focus on analysis
            cleaned_query = self._clean_query_for_pandas_agent(analysis.paraphrased_query, analysis.intent)
            print("Clean Query to PandasAgent:", cleaned_query)
            logger.info(f"Sending cleaned query to PandasAgent: {cleaned_query}")
            
            # Call PandasAgent with CLEAN, FOCUSED query
            result = await self.pandas_agent.process_query(cleaned_query)
            
            # Extract components
            answer = result.get('answer', 'No results found.')
            visualization_html = result.get('visualization', None)
            metadata = result.get('metadata', {})
            
            # Process visualization if available
            if visualization_html:
                logger.info(f"Visualization generated, HTML length: {len(visualization_html)}")
                
                # Clean up visualization HTML
                if '<html' in visualization_html.lower() or '<!doctype' in visualization_html.lower():
                    import re
                    body_match = re.search(r'<body[^>]*>(.*?)<\/body>', visualization_html, re.DOTALL | re.IGNORECASE)
                    if body_match:
                        visualization_html = body_match.group(1)
                
                # Ensure HTTPS for external sources
                visualization_html = visualization_html.replace('http://', 'https://')
            else:
                logger.info("No visualization generated for data analysis")
            
            return {
                "response": answer,
                "visualization_html": visualization_html,
                "has_visualization": bool(visualization_html),
                "agent_used": "PandasAgent",
                "metadata": metadata
            }
                
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return {
                "response": f"I encountered an error analyzing the data: {str(e)}. Please try rephrasing your query or check if the requested data exists in our database.",
                "visualization_html": None,
                "has_visualization": False,
                "agent_used": "PandasAgent (Error)",
                "metadata": {"error": str(e)}
            }
    
    # def _clean_query_for_pandas_agent(self, query: str, intent: str) -> str:
    #     """Clean and focus query for PandasAgent to get better results"""
        
    #     # Remove conversational fluff and focus on the analytical request
    #     query_lower = query.lower()
        
    #     # Handle patient details queries specifically
    #     if intent == "patient_details" or ('detailed information' in query_lower and 'patient id' in query_lower):
    #         # Patient details query - extract patient ID and make it focused
    #         import re
    #         patient_id_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', query)
    #         if patient_id_match:
    #             patient_id = patient_id_match.group(1)
    #             return f"Show comprehensive patient information for patient ID {patient_id} including demographics, age, gender, location, TBI incident details, medical history, and any available background data from all relevant tables."
        
    #     # For general data analysis queries, clean up conversational elements
    #     cleaned_query = query
        
    #     # Remove conversational starters that add noise
    #     conversational_removals = [
    #         "I want detailed information from the database about ",
    #         "I want to analyze ",
    #         "Can you show me ",
    #         "Please provide ",
    #         "I would like to see ",
    #         "Based on the available data, ",
    #         "In our database, "
    #     ]
        
    #     for removal in conversational_removals:
    #         if cleaned_query.lower().startswith(removal.lower()):
    #             cleaned_query = cleaned_query[len(removal):]
        
    #     # Clean up redundant phrases that don't add analytical value
    #     redundant_phrases = [
    #         "based on the available data",
    #         "in our database",
    #         "from the database",
    #         "that we have"
    #     ]
        
    #     for phrase in redundant_phrases:
    #         cleaned_query = cleaned_query.replace(phrase, "").replace(phrase.title(), "")
        
    #     # Focus on the core analytical request
    #     if 'patient' in cleaned_query.lower() and ('most' in cleaned_query.lower() or 'highest' in cleaned_query.lower()):
    #         # This is asking about identifying top patients - keep it focused
    #         if 'symptom logs' in cleaned_query.lower():
    #             cleaned_query = "Which patients have the highest number of symptom logs? Show top 10 with counts and percentages."
    #     elif 'demographics' in cleaned_query.lower() and 'patient id' in cleaned_query.lower():
    #         # Patient demographic request - already handled above
    #         pass
        
    #     # Clean up extra whitespace and ensure proper formatting
    #     cleaned_query = ' '.join(cleaned_query.split())
        
    #     # Ensure it ends with a clear analytical request
    #     if not cleaned_query.strip().endswith('?'):
    #         cleaned_query = cleaned_query.strip() + "?"
        
    #     logger.info(f"Cleaned query: '{query}' → '{cleaned_query}'")
    #     return cleaned_query.strip()
    
    async def _handle_medical_query(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle general medical queries"""
        try:
            if self.medical_agent:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.medical_agent.process_query(analysis.paraphrased_query)
                )
                
                # Handle different response types
                if isinstance(response, dict):
                    text_response = response.get('response', response.get('answer', response.get('text', str(response))))
                elif isinstance(response, str):
                    text_response = response
                else:
                    text_response = str(response)
                
                return {
                    "response": text_response,
                    "agent_used": "Medical Agent"
                }
            else:
                return {
                    "response": f"I understand you're asking about {analysis.paraphrased_query}. While I don't have access to the full medical database right now, I recommend consulting medical literature or healthcare professionals for specific medical information.",
                    "agent_used": "Medical Agent (Limited)"
                }
        except Exception as e:
            logger.error(f"Medical agent error: {e}")
            return {
                "response": "I'm experiencing technical difficulties accessing medical information. Please try rephrasing your question or consult medical literature.",
                "agent_used": "Medical Agent (Error)"
            }
    
    async def _handle_tbi_knowledge(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle TBI knowledge queries with fallback to MedPalm"""
        primary_response = ""
        fallback_used = False
        
        try:
            # Try TBI Retrieval Agent first
            if self.retrieval_agent:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.retrieval_agent.ask_question(analysis.paraphrased_query, top_k=8)
                )
                
                # Handle different response types
                if isinstance(result, dict):
                    primary_response = result.get('answer', result.get('response', str(result)))
                else:
                    primary_response = str(result)
                
                # Check if response is insufficient
                insufficient_indicators = [
                    "I don't have specific information",
                    "limited information",
                    "not available in my database",
                    "recommend checking other sources",
                    "difficult with the information I currently have",
                    len(primary_response.strip()) < 100
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
                            
                            # Process MedPalm response
                            if isinstance(fallback_response, dict):
                                primary_response = fallback_response.get('response', fallback_response.get('answer', str(fallback_response)))
                            else:
                                primary_response = str(fallback_response)
                            
                            fallback_used = True
                            logger.info("✅ MedPalm fallback successful for TBI query")
                        except Exception as e:
                            logger.error(f"❌ MedPalm fallback failed: {e}")
                            # Keep original TBI response if fallback fails
                
            else:
                # No TBI agent, go directly to MedPalm
                if self.medical_agent:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: self.medical_agent.process_query(analysis.paraphrased_query)
                    )
                    
                    if isinstance(response, dict):
                        primary_response = response.get('response', response.get('answer', str(response)))
                    else:
                        primary_response = str(response)
                    
                    fallback_used = True
                else:
                    primary_response = f"I understand you're asking about TBI: {analysis.paraphrased_query}. While I don't have access to the TBI database right now, I recommend consulting medical literature for TBI-specific information."
        
        except Exception as e:
            logger.error(f"TBI query handling error: {e}")
            primary_response = "I'm experiencing technical difficulties accessing TBI information. Please try rephrasing your question."
        
        return {
            "response": primary_response,
            "agent_used": "MedPalm (Fallback)" if fallback_used else "TBI Specialist",
            "fallback_used": fallback_used
        }
    
    async def _handle_schema_info(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle database schema information queries"""
        try:
            query_lower = analysis.paraphrased_query.lower()
            
            if 'how many table' in query_lower:
                response = """We have **7 main tables** in our TBI database:

1. **patients** - Core patient demographics and information
2. **tbi_incidents** - TBI incident details and causes  
3. **symptom_logs** - Longitudinal symptom tracking
4. **worst_symptoms** - Patient-reported worst symptoms
5. **therapies** - Treatment and therapy information
6. **social_determinants** - Social factors affecting health
7. **symptom_reference** - Reference data for symptoms

Each table contains specific columns for comprehensive TBI research. Would you like details about any specific table?"""
            else:
                # Use LLM for more complex schema queries
                schema_prompt = f"""
You are helping a researcher understand the TBI database schema.

AVAILABLE SCHEMA INFORMATION:
{DB_SCHEMA}

DETAILED SCHEMA ANALYSIS:
{getattr(self, 'schema_info', 'Schema analysis not available')}

RESEARCHER QUESTION: "{analysis.paraphrased_query}"

Provide a clear, detailed response about the database schema, focusing on:
1. Relevant tables and their purpose
2. Key columns and data types
3. Relationships between tables
4. Data quality considerations
5. Suggestions for analysis approaches

Keep the response informative but concise.

Response:"""

                loop = asyncio.get_event_loop()
                llm_response = await loop.run_in_executor(
                    None,
                    lambda: self.gemini_model.generate_content(
                        schema_prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                )
                
                response = llm_response.text
            
            return {
                "response": response,
                "agent_used": "Schema Info"
            }
            
        except Exception as e:
            logger.error(f"Error handling schema query: {e}")
            return {
                "response": f"Here's the basic schema information:\n\n{DB_SCHEMA}",
                "agent_used": "Schema Info (Fallback)"
            }
    
    async def _handle_methodology(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle research methodology queries"""
        try:
            methodology_prompt = f"""
You are a research methodology advisor for TBI data analysis.

RESEARCHER QUESTION: "{analysis.paraphrased_query}"

AVAILABLE DATA CONTEXT:
- TBI patient database with incidents, symptoms, therapies, and social determinants
- Demographic information including age, location, veteran status
- Temporal data on incidents and symptom progression
- Categorical and numerical variables

Provide expert guidance on:
1. Appropriate statistical methods for the research question
2. Data preparation considerations
3. Potential confounding variables
4. Interpretation guidelines
5. Limitations and caveats

Be specific and practical in your recommendations.

Response:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    methodology_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            return {
                "response": response.text,
                "agent_used": "Methodology Advisor"
            }
            
        except Exception as e:
            logger.error(f"Error handling methodology query: {e}")
            return {
                "response": "I encountered an error providing methodology guidance. For statistical analysis of TBI data, consider consulting with a biostatistician or research methodology literature.",
                "agent_used": "Methodology Advisor (Error)"
            }
    
    async def _handle_patient_details(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle specific patient detail requests by routing to PandasAgent with FOCUSED queries"""
        try:
            if not self.pandas_agent:
                return {
                    "response": "Data analysis agent is not available. Please check system configuration.",
                    "visualization_html": None,
                    "has_visualization": False,
                    "agent_used": "PandasAgent (Unavailable)",
                    "metadata": {"error": "PandasAgent not initialized"}
                }
            
            # CRITICAL: Clean the query for PandasAgent - focus on patient details retrieval
            cleaned_query = self._clean_query_for_pandas_agent(analysis.paraphrased_query, "patient_details")
            
            logger.info(f"Sending focused patient details query to PandasAgent: {cleaned_query}")
            
            # Pass cleaned query to PandasAgent for database querying
            result = await self.pandas_agent.process_query(cleaned_query)
            
            # Extract components
            answer = result.get('answer', 'No patient information found.')
            visualization_html = result.get('visualization', None)
            metadata = result.get('metadata', {})
            
            # Process visualization if available
            if visualization_html:
                logger.info(f"Patient details visualization generated, HTML length: {len(visualization_html)}")
                
                # Clean up visualization HTML
                if '<html' in visualization_html.lower() or '<!doctype' in visualization_html.lower():
                    import re
                    body_match = re.search(r'<body[^>]*>(.*?)<\/body>', visualization_html, re.DOTALL | re.IGNORECASE)
                    if body_match:
                        visualization_html = body_match.group(1)
                
                # Ensure HTTPS for external sources
                visualization_html = visualization_html.replace('http://', 'https://')
            else:
                logger.info("No visualization generated for patient details")
            
            return {
                "response": answer,
                "visualization_html": visualization_html,
                "has_visualization": bool(visualization_html),
                "agent_used": "PandasAgent (Patient Details)",
                "metadata": metadata
            }
                
        except Exception as e:
            logger.error(f"Error in patient details query: {e}")
            return {
                "response": f"I encountered an error retrieving patient details: {str(e)}. Please ensure the patient ID is valid and try again.",
                "visualization_html": None,
                "has_visualization": False,
                "agent_used": "PandasAgent (Error)",
                "metadata": {"error": str(e)}
            }
    
    async def _handle_continuation_query(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle simple continuation queries (yes/no/thanks responses)"""
        try:
            # Get full conversation context with data
            conversation_context = self.memory.get_recent_context(3)
            
            # Use the paraphrased query which should contain specific data references
            continuation_prompt = f"""
You are a research assistant continuing a conversation with a researcher. You have access to the complete conversation history including all data tables and specific values.

FULL CONVERSATION CONTEXT:
{conversation_context}

RESEARCHER'S CONTINUATION QUERY: "{analysis.original_query}"
PARAPHRASED QUERY (with specific data): "{analysis.paraphrased_query}"

TASK: Provide a direct, specific answer using the ACTUAL data from the conversation context above.

CRITICAL INSTRUCTIONS:
1. Use ACTUAL patient IDs, numbers, and values from the conversation history
2. Do NOT use placeholders like "[Insert Patient ID]" or "[Insert Number]"
3. Extract specific data points from previous responses
4. If the previous response contained a table, reference the exact values from that table
5. Be specific and helpful with real data

For example, if the conversation shows a table with:
| Patient ID | Number of Symptom Logs | Percentage |
| 1ba40009-06c2-4ef6-912d-89123343015f | 9354 | 21.14 |

And the researcher asks "who is that patient?", you should respond with the actual patient ID and details.

Provide a direct, helpful answer using the real data from the conversation context:
"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    continuation_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            return {
                "response": response.text.strip(),
                "agent_used": "Continuation Handler"
            }
            
        except Exception as e:
            logger.error(f"Error handling continuation query: {e}")
            return {
                "response": "I understand you're referring to our previous conversation, but I'm having trouble accessing the specific data. Could you please rephrase your question more specifically?",
                "agent_used": "Continuation Handler (Error)"
            }
    
    async def _handle_platform_info(self, analysis: ResearchQueryAnalysis) -> Dict[str, Any]:
        """Handle platform and identity questions"""
        query_lower = analysis.paraphrased_query.lower()
        
        if any(keyword in query_lower for keyword in ['power of patients', 'what is power of patients']):
            response = self.company_info
        elif any(keyword in query_lower for keyword in ['who are you', 'who is sallie', 'tell me about yourself']):
            response = f"""I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. 

I'm designed to help researchers like you with:
• TBI data analysis and visualization
• Medical information queries
• Research methodology guidance
• Understanding our comprehensive TBI database

{self.company_info}

How can I assist you with your research today?"""
        else:
            response = f"I understand you're asking: {analysis.paraphrased_query}. I'm Sallie, your research assistant from Power of Patients. I'm here to help with TBI research, data analysis, and medical information. What would you like to explore?"
        
        return {
            "response": response,
            "agent_used": "Platform Info"
        }
    
    async def _generate_contextual_response(self, agent_result: Dict[str, Any], analysis: ResearchQueryAnalysis) -> str:
        """Generate final contextual response with SMART context usage - avoid bleeding unrelated previous data"""
        
        # Get conversation context with FULL data
        conversation_history = self.memory.get_recent_context(3)
        has_conversation_history = len(self.memory.exchanges) > 0
        researcher_id = self.memory.current_researcher_id
        
        # Research disclaimer
        research_disclaimer = ""
        if analysis.intent in ["medical_general", "tbi_knowledge"]:
            research_disclaimer = """

📚 **Research Note:** This information is provided for research and educational purposes. For clinical applications or patient care decisions, always refer to peer-reviewed literature and clinical guidelines."""
        
        # Check if response contains HTML tables
        agent_response = agent_result["response"]
        contains_html_tables = '<table' in agent_response
        
        # CRITICAL: Determine if current query is related to previous conversation using LLM
        is_query_related_to_previous = await self._is_query_contextually_related(analysis, conversation_history)
        
        # For continuation queries, if the agent already provided specific data, use it directly
        if analysis.intent == "continuation" and "patient" in analysis.paraphrased_query.lower():
            return agent_response + research_disclaimer
        
        # For completely unrelated queries (like general medical questions), don't include previous patient data
        if not is_query_related_to_previous:
            logger.info(f"Query '{analysis.original_query}' is unrelated to previous conversation - using clean response")
            return agent_response + research_disclaimer
        
        # Build context-aware prompt with EMPHASIS on preserving specific data (only for related queries)
        response_prompt = f"""
You are Sallie, a professional research assistant. Generate a natural, contextual response with SPECIFIC data preservation.

CRITICAL CONTEXT RULES:
- CURRENT QUERY IS RELATED TO PREVIOUS CONVERSATION: {'YES' if is_query_related_to_previous else 'NO'}
- ONLY reference previous data if the current query is directly related to it
- If current query is about a completely different topic, DO NOT mention previous patient data
- CONVERSATION HISTORY EXISTS: {'YES' if has_conversation_history else 'NO'}

RESEARCHER INFORMATION:
- ID: {researcher_id}
- Current Query: "{analysis.original_query}"
- Paraphrased Query: "{analysis.paraphrased_query}"
- Intent: {analysis.intent}

{'RELEVANT CONVERSATION HISTORY:' if is_query_related_to_previous and has_conversation_history else 'CONVERSATION HISTORY: Not relevant to current query'}
{conversation_history if is_query_related_to_previous and has_conversation_history else 'Current query is unrelated to previous conversation.'}

SPECIALIST AGENT RESPONSE:
Agent Used: {agent_result.get('agent_used', 'Unknown')}
Response: {agent_response}

{'THE RESPONSE CONTAINS HTML TABLES. YOU MUST:' if contains_html_tables else ''}
{'1. Provide natural language summary before tables with SPECIFIC numbers' if contains_html_tables else ''}
{'2. Keep all HTML table tags EXACTLY as they are' if contains_html_tables else ''}
{'3. Reference ACTUAL patient IDs and values from tables' if contains_html_tables else ''}

RESPONSE GUIDELINES:
1. Address the researcher professionally
2. {'Use ACTUAL data values from conversation history - never use placeholders' if is_query_related_to_previous else 'Provide a clean response focused on the current query without referencing unrelated previous data'}
3. {'If previous response had specific patient data, use those exact values' if is_query_related_to_previous else 'Do NOT reference any previous patient data as it is unrelated to this query'}
4. Build upon specialist agent's response appropriately
5. For data analysis, highlight key findings with appropriate context
6. Keep response natural and conversational for research context
7. End with appropriate follow-up question or research guidance
8. {'CRITICAL: Extract and use real data values from conversation history' if is_query_related_to_previous else 'CRITICAL: Focus only on the current query topic without mentioning unrelated previous conversations'}
{'9. CRITICAL: Preserve all HTML table structures exactly' if contains_html_tables else ''}

EXAMPLE OF PROPER CONTEXT USAGE:
- If current query is "tell me about diabetes" and previous was about patient X, DO NOT mention patient X
- If current query is "show me that patient's demographics" and previous identified patient Y, DO mention patient Y
- If current query is "what causes TBI" and previous was data analysis, DO NOT reference specific previous data

Generate a research-appropriate response {'with specific data context' if is_query_related_to_previous else 'focused on the current query only'}:"""

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
            
            final_response = response.text.strip()
            
            # Preserve HTML tables if they were lost
            if contains_html_tables and '<table' not in final_response:
                logger.warning("HTML tables were lost during formatting, using original response")
                final_response = agent_response
            
            # Check if response still contains placeholders - if so, use agent response directly
            if "[Insert" in final_response or "Insert Patient ID" in final_response:
                logger.warning("Response contains placeholders, using agent response directly")
                final_response = agent_response
            
            # CRITICAL: Check for context bleeding in unrelated queries using LLM
            if not is_query_related_to_previous and await self._contains_patient_references(final_response):
                logger.warning("LLM detected context bleeding in unrelated query, using clean agent response")
                final_response = agent_response
            
            # Add research disclaimer
            final_response += research_disclaimer
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {e}")
            return agent_response + research_disclaimer
    
    async def _is_query_contextually_related(self, analysis: ResearchQueryAnalysis, conversation_history: str) -> bool:
        """Use LLM to determine if current query is related to previous conversation context"""
        
        if not conversation_history.strip():
            return False
        
        # Always related for continuation queries
        if analysis.intent == "continuation":
            return True
        
        json_template = """{
    "is_related": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of why query is or isn't related"
}"""

        context_analysis_prompt = f"""
You are analyzing whether a researcher's current query is related to their previous conversation context.

PREVIOUS CONVERSATION CONTEXT:
{conversation_history}

CURRENT QUERY: "{analysis.original_query}"
PARAPHRASED QUERY: "{analysis.paraphrased_query}"
QUERY INTENT: {analysis.intent}

TASK: Determine if the current query is contextually related to the previous conversation.

GUIDELINES:
- If query references specific patients, data, or results from previous conversation → RELATED
- If query is a follow-up question about previous analysis → RELATED  
- If query asks for details about something mentioned before → RELATED
- If query is a completely different topic (general medical questions, new analysis) → NOT RELATED
- If query is general knowledge (diabetes, TBI info) unrelated to previous specific data → NOT RELATED

EXAMPLES:
- Previous: "Patient X has 1000 symptom logs" → Current: "tell me about diabetes" → NOT RELATED
- Previous: "Patient X has 1000 symptom logs" → Current: "show me that patient's details" → RELATED
- Previous: "Age distribution analysis" → Current: "what causes TBI" → NOT RELATED
- Previous: "Patient Y analysis" → Current: "more information about them" → RELATED

Respond in JSON format:
{json_template}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    context_analysis_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
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
                is_related = result.get("is_related", False)
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "No reasoning provided")
                
                logger.info(f"Context analysis: Query is {'RELATED' if is_related else 'NOT RELATED'} (confidence: {confidence:.2f}) - {reasoning}")
                return is_related
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse context analysis JSON: {result_text}")
                return False
        
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return False
    
    async def _clean_query_for_pandas_agent(self, query: str, intent: str) -> str:
        """Use LLM to clean and focus query for PandasAgent"""
        
        json_template = """{
    "cleaned_query": "focused analytical query for database analysis",
    "reasoning": "explanation of changes made"
}"""

        query_cleaning_prompt = f"""
You are optimizing queries for a data analysis agent (PandasAgent) that works with TBI research databases.

ORIGINAL QUERY: "{query}"
QUERY INTENT: {intent}

TASK: Transform this query into a focused, analytical query that will get better results from the data analysis agent.

OPTIMIZATION GUIDELINES:
1. Remove conversational fluff ("I want", "Can you", "Please show me")
2. Remove redundant context ("from the database", "based on available data")
3. Focus on the core analytical request
4. For patient details: Make it specific about what demographic/medical info to retrieve
5. For data analysis: Focus on the specific analysis or visualization needed
6. Keep it concise but complete
7. Ensure it's a clear, actionable query for database analysis

EXAMPLES:
- "I want detailed information about patient ID 123 who had lots of symptom logs" → "Show comprehensive patient information for patient ID 123 including demographics, medical history, TBI details, and symptom data"
- "Can you please show me which patients have the most symptom logs in our database?" → "Which patients have the highest number of symptom logs? Show top 10 with counts and percentages"
- "I would like to analyze the distribution of ages by gender" → "Show age distribution by gender with statistical summary and visualization"

IMPORTANT: Keep the core meaning and all important identifiers (patient IDs, dates, etc.) but make it focused for data analysis.

Respond in JSON format:
{json_template}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    query_cleaning_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
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
                cleaned_query = result.get("cleaned_query", query)
                reasoning = result.get("reasoning", "No reasoning provided")
                
                logger.info(f"Query cleaning: '{query}' → '{cleaned_query}' | Reasoning: {reasoning}")
                return cleaned_query
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse query cleaning JSON: {result_text}")
                return query
        
        except Exception as e:
            logger.error(f"Query cleaning failed: {e}")
            return query
    
    async def _contains_patient_references(self, response: str) -> bool:
        """Use LLM to detect if response contains specific patient references that indicate context bleeding"""
        
        json_template = """{
    "contains_patient_refs": true/false,
    "confidence": 0.0-1.0,
    "detected_references": ["list of specific patient references found"],
    "reasoning": "explanation"
}"""

        reference_detection_prompt = f"""
You are analyzing a response to detect if it contains specific patient references that might indicate inappropriate context bleeding.

RESPONSE TO ANALYZE: "{response}"

TASK: Determine if this response contains specific patient references that suggest it's inappropriately referencing previous patient data.

WHAT TO DETECT:
✅ PATIENT REFERENCES (indicate context bleeding):
- Specific patient IDs (UUID format)
- References to "patient we analyzed", "previous patient", "Tyler Sasser"
- Specific symptom log counts with percentages
- References to specific patients by name or characteristics

❌ NOT PATIENT REFERENCES (these are okay):
- General medical information about diseases
- Generic references to "patients" in general
- Statistical information without specific patient identifiers
- Educational content about medical conditions

EXAMPLES:
- "Diabetes affects many patients..." → NO patient references
- "Patient 123-456-789 had 1000 logs..." → YES, contains specific patient reference
- "Tyler Sasser who has 20556 symptom logs..." → YES, contains specific patient reference
- "Understanding diabetes symptoms..." → NO patient references

Respond in JSON format:
{json_template}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    reference_detection_prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
            )
            
            # Parse JSON response
            result_text = llm_response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            try:
                result = json.loads(result_text)
                contains_refs = result.get("contains_patient_refs", False)
                confidence = result.get("confidence", 0.5)
                detected_refs = result.get("detected_references", [])
                reasoning = result.get("reasoning", "No reasoning provided")
                
                if contains_refs:
                    logger.warning(f"Patient references detected in response: {detected_refs} | Confidence: {confidence:.2f} | Reasoning: {reasoning}")
                else:
                    logger.info(f"No patient references detected | Confidence: {confidence:.2f}")
                
                return contains_refs
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse reference detection JSON: {result_text}")
                return False
        
        except Exception as e:
            logger.error(f"Reference detection failed: {e}")
            return False
    
    async def process_query(self, query: str, researcher_id: str = "default") -> Dict[str, Any]:
        """
        Main query processing with LLM-powered intelligence
        """
        start_time = datetime.now()
        
        try:
            # Handle empty query - treat as INITIAL_CONNECTION
            if not query or query == 'INITIAL_CONNECTION':
                query = ''
            
            # Set current researcher
            self.memory.current_researcher_id = researcher_id
            
            # Check if this is a new researcher
            if researcher_id not in self.welcomed_researchers:
                self.welcomed_researchers.add(researcher_id)
                
                # If this is just an initial connection, return welcome only
                if not query:
                    processing_time = (datetime.now() - start_time).total_seconds()
                    welcome_message = self._generate_researcher_welcome(researcher_id)
                    
                    # Store welcome in conversation memory
                    self.memory.add_exchange("Initial connection", welcome_message, "Welcome Service")
                    
                    return {
                        "success": True,
                        "researcher_id": researcher_id,
                        "query": "Initial connection",
                        "intent": "welcome_message",
                        "response": welcome_message,
                        "visualization_html": None,
                        "has_visualization": False,
                        "processing_time": processing_time,
                        "metadata": {"event": "welcome"},
                        "is_welcome_message": True
                    }
            
            # If query is still empty, use default
            if not query:
                query = "Hello"
            
            # Step 1: Analyze query with LLM intelligence
            logger.info(f"Analyzing research query: {query[:50]}...")
            analysis = await self.query_analyzer.analyze_research_query(query, self.memory)
            
            # Step 2: Guardrail check (permissive for research)
            conversation_context = self.memory.get_recent_context(3)
            guardrail_result = await self.guardrail_agent.analyze_research_query_safety(
                analysis.paraphrased_query, conversation_context
            )
            
            if not guardrail_result["allow"] and guardrail_result["confidence"] > 0.95:
                # Very restrictive - only block clearly inappropriate content
                processing_time = (datetime.now() - start_time).total_seconds()
                
                self.memory.add_exchange(query, guardrail_result["redirect_message"], "Guardrail Agent")
                
                return {
                    "success": True,
                    "researcher_id": researcher_id,
                    "query": query,
                    "paraphrased_query": analysis.paraphrased_query,
                    "intent": "blocked",
                    "response": guardrail_result["redirect_message"],
                    "visualization_html": None,
                    "has_visualization": False,
                    "processing_time": processing_time,
                    "guardrail_blocked": True
                }
            
            # Step 3: Route to appropriate agent based on analysis
            logger.info(f"Routing research query with intent: {analysis.intent}")
            
            agent_result = {}
            
            if analysis.intent == "data_analysis":
                # Check if this is specifically about patient details
                patient_detail_keywords = ['details', 'demographics', 'information', 'profile', 'background', 'about patient']
                is_patient_detail_query = any(keyword in analysis.paraphrased_query.lower() for keyword in patient_detail_keywords)
                
                if is_patient_detail_query:
                    logger.info("Detected patient details query, routing to specialized handler")
                    agent_result = await self._handle_patient_details(analysis)
                else:
                    logger.info("Standard data analysis query")
                    agent_result = await self._handle_data_analysis(analysis)
            elif analysis.intent == "continuation":
                agent_result = await self._handle_continuation_query(analysis)
            elif analysis.intent == "medical_general":
                agent_result = await self._handle_medical_query(analysis)
            elif analysis.intent == "tbi_knowledge":
                agent_result = await self._handle_tbi_knowledge(analysis)
            elif analysis.intent == "schema_info":
                agent_result = await self._handle_schema_info(analysis)
            elif analysis.intent == "methodology":
                agent_result = await self._handle_methodology(analysis)
            elif analysis.intent == "platform_info":
                agent_result = await self._handle_platform_info(analysis)
            else:
                # General conversation
                agent_result = {
                    "response": f"I understand you're asking: {analysis.paraphrased_query}. I'm here to help with TBI research, data analysis, and medical information. What would you like to explore?",
                    "agent_used": "General Conversation"
                }
            
            # Step 4: Generate final contextual response
            final_response = await self._generate_contextual_response(agent_result, analysis)
            
            # Step 5: Update conversation memory
            metadata = {
                "intent": analysis.intent,
                "agent_used": agent_result.get("agent_used", "Unknown"),
                "confidence": analysis.confidence,
                "fallback_used": agent_result.get("fallback_used", False)
            }
            
            if agent_result.get("metadata"):
                metadata.update(agent_result["metadata"])
            
            self.memory.add_exchange(query, final_response, agent_result.get("agent_used", "Unknown"), metadata)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Research query processed successfully in {processing_time:.2f}s")
            
            return {
                "success": True,
                "researcher_id": researcher_id,
                "query": query,
                "paraphrased_query": analysis.paraphrased_query if analysis.paraphrased_query != query else None,
                "intent": analysis.intent,
                "response": final_response,
                "visualization_html": agent_result.get("visualization_html"),
                "has_visualization": agent_result.get("has_visualization", False),
                "processing_time": processing_time,
                "confidence": analysis.confidence,
                "urgency_level": analysis.urgency_level,
                "analysis_reasoning": analysis.reasoning,
                "fallback_used": agent_result.get("fallback_used", False)
            }
            
        except Exception as e:
            logger.exception(f"Error processing research query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your research request. Please try again or rephrase your query.",
                "processing_time": processing_time
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive research session summary"""
        # Extract analysis statistics
        analysis_types = set()
        visualizations_created = 0
        agents_used = set()
        
        for exchange in self.memory.exchanges:
            metadata = exchange.get('metadata', {})
            if metadata.get('intent'):
                analysis_types.add(metadata['intent'])
            if metadata.get('agent_used'):
                agents_used.add(metadata['agent_used'])
            if exchange.get('response') and '<table' in exchange['response']:
                visualizations_created += 1
        
        # Count query intents
        query_intents = {}
        for exchange in self.memory.exchanges:
            intent = exchange.get('metadata', {}).get('intent', 'unknown')
            query_intents[intent] = query_intents.get(intent, 0) + 1
        
        return {
            "researcher_id": self.memory.current_researcher_id,
            "session_length": len(self.memory.exchanges),
            "analyses_performed": query_intents.get('data_analysis', 0),
            "visualizations_created": visualizations_created,
            "medical_queries": query_intents.get('medical_general', 0) + query_intents.get('tbi_knowledge', 0),
            "agents_used": list(agents_used),
            "query_breakdown": query_intents,
            "total_processing_time": sum([exchange.get('metadata', {}).get('processing_time', 0) for exchange in self.memory.exchanges]),
            "database_connected": bool(self.pandas_agent and hasattr(self.pandas_agent, 'dataframes')),
            "available_agents": {
                "pandas_agent": self.pandas_agent is not None,
                "medical_agent": self.medical_agent is not None,
                "retrieval_agent": self.retrieval_agent is not None
            },
            "fallback_usage": sum([1 for exchange in self.memory.exchanges if exchange.get('metadata', {}).get('fallback_used', False)]),
            "memory_usage": f"{len(self.memory.exchanges)}/{self.memory.exchanges.maxlen}"
        }
    
    def clear_session(self):
        """Clear current research session"""
        self.memory.exchanges.clear()
        self.memory.last_agent_used = ""
        self.memory.pending_followup = None
        self.memory.analysis_history.clear()
        self.welcomed_researchers.clear()
        logger.info("Research session cleared")
    
    def get_available_analyses(self) -> Dict[str, List[str]]:
        """Get categorized list of available research analyses"""
        return {
            "data_analysis": [
                "What are the most common causes of TBI in our patient population?",
                "Show the age distribution of TBI patients by gender",
                "What are the most frequent immediate symptoms after TBI?",
                "Which states have the highest number of TBI patients?",
                "What's the relationship between injury location and symptom severity?",
                "Which therapies are most commonly prescribed for TBI patients?",
                "How does veteran status relate to TBI causes?",
                "What percentage of patients had TBI before their current incident?",
                "Show symptom categories by frequency and average severity"
            ],
            "patient_details": [
                "Show me details of patient [patient_id]",
                "What are the demographics of that patient?",
                "I want to see the full profile of the top patient",
                "Give me background information on patient [patient_id]",
                "Show patient information for the highest symptom logger"
            ],
            "medical_queries": [
                "What are the typical symptoms of mild TBI?",
                "How is post-concussion syndrome diagnosed?",
                "What medications are used for TBI-related headaches?",
                "Explain the Glasgow Coma Scale",
                "What is the difference between a concussion and a contusion?"
            ],
            "methodology": [
                "What statistical test should I use to compare TBI outcomes by gender?",
                "How can I control for age as a confounding variable?",
                "What's the best way to visualize symptom progression over time?"
            ],
            "context_examples": [
                "✅ LLM Context Detection: 'Tell me about that patient' (after showing patient data) → Uses specific context",
                "✅ LLM Clean Response: 'Tell me about diabetes' (general topic) → Clean medical info without previous data", 
                "✅ LLM Query Optimization: 'I want details about patient X' → 'Show comprehensive patient information for patient X'",
                "✅ Natural Language: Handles any phrasing - 'give me info on that person', 'what about them', etc.",
                "❌ No Keywords: System understands intent regardless of exact words used"
            ]
        }

# Terminal Testing Interface
async def terminal_interface():
    """Enhanced terminal interface for testing Enhanced ResearcherAgent"""
    print("=" * 80)
    print("🧠 ENHANCED PROFESSIONAL RESEARCHER AGENT - TERMINAL INTERFACE")
    print("=" * 80)
    
    # Initialize agent
    print("Initializing Enhanced Researcher Agent...")
    agent = ProfessionalResearcherAgent()
    
    # Connect to database
    print("Connecting to TBI database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database")
        return
    
    print("✅ Enhanced Researcher Agent initialized successfully!")
    print("\n💡 Enhanced Features:")
    print("   • LLM-powered query analysis and paraphrasing")
    print("   • Intelligent research-focused guardrail system")
    print("   • Smart conversation memory management")
    print("   • TBI → MedPalm fallback system") 
    print("   • Advanced intent classification for research")
    print("   • LLM-powered context-aware response generation")
    print("   • 🎯 LLM-based context bleeding prevention")
    print("   • 🔧 LLM-optimized pandas queries for better results")
    print("   • 🧠 No keyword limitations - handles natural language variations")
    
    # Show available agents
    summary = agent.get_session_summary()
    print("\n🔧 Available Agents:")
    for agent_name, available in summary['available_agents'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {agent_name}")
    
    print("\n📋 Commands:")
    print("   • Enter research queries naturally - intelligent analysis and routing")
    print("   • 'examples' - View example queries by category (including context examples)")
    print("   • 'analyze <query>' - See detailed query analysis")
    print("   • 'schema' - View database schema")
    print("   • 'summary' - View session summary")
    print("   • 'clear' - Clear session")
    print("   • 'quit' - Exit")
    print("\n🎯 LLM Intelligence:")
    print("   • Understands natural language variations (not limited by keywords)")
    print("   • General topics (diabetes, TBI info) → Clean responses")
    print("   • Patient references (that patient, show details) → Use specific data")
    print("   • LLM-optimized pandas queries → Better analysis results")
    print("   • Smart context detection → Prevents inappropriate data bleeding")
    print("-" * 80)
    
    researcher_id = input("\nEnter your researcher ID (or press Enter for 'default'): ").strip() or "default"
    
    while True:
        try:
            user_input = input(f"\n[Researcher {researcher_id}] Query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.clear_session()
                print("✅ Session cleared")
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_session_summary()
                print("\n📊 SESSION SUMMARY:")
                print(json.dumps(summary, indent=2, default=str))
                continue
            elif user_input.lower() == 'examples':
                print("\n📝 EXAMPLE QUERIES BY CATEGORY:")
                examples = agent.get_available_analyses()
                for category, queries in examples.items():
                    print(f"\n{category.upper().replace('_', ' ')}:")
                    for i, example in enumerate(queries[:5], 1):
                        if category == 'context_examples':
                            print(f"  {example}")  # Don't number these
                        else:
                            print(f"  {i}. {example}")
                continue
            elif user_input.lower().startswith('analyze '):
                analyze_query = user_input[8:].strip()
                if analyze_query:
                    print("🔍 Analyzing research query...")
                    analysis = await agent.query_analyzer.analyze_research_query(analyze_query, agent.memory)
                    print(f"\n🧠 RESEARCH QUERY ANALYSIS:")
                    print(f"   Original: {analysis.original_query}")
                    print(f"   Paraphrased: {analysis.paraphrased_query}")
                    print(f"   Intent: {analysis.intent} (confidence: {analysis.confidence:.2f})")
                    print(f"   Is Data Query: {analysis.is_data_query}")
                    print(f"   Requires Visualization: {analysis.requires_visualization}")
                    print(f"   Suggested Agent: {analysis.suggested_agent}")
                    print(f"   Urgency: {analysis.urgency_level}")
                    print(f"   Reasoning: {analysis.reasoning}")
                continue
            elif user_input.lower() == 'schema':
                print("\n📋 DATABASE SCHEMA:")
                print(getattr(agent, 'schema_info', DB_SCHEMA))
                continue
            
            # Process query
            print("🤖 Processing research query...")
            result = await agent.process_query(user_input, researcher_id)
            
            if result["success"]:
                print(f"\n💬 Sallie: {result['response']}")
                
                # Show visualization info
                if result.get('has_visualization') and result.get('visualization_html'):
                    print("\n📈 VISUALIZATION GENERATED:")
                    print("(Interactive chart would appear in web interface)")
                    print(f"HTML Length: {len(result['visualization_html'])} characters")
                    
                    # Option to save visualization
                    save_viz = input("\nSave visualization to HTML file? (y/n): ").strip().lower()
                    if save_viz == 'y':
                        filename = f"research_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(result['visualization_html'])
                        print(f"✅ Visualization saved to: {filename}")
                
                # Show analysis details
                print(f"\n📊 Analysis:")
                print(f"   Intent: {result['intent']}")
                print(f"   Agent Used: {result.get('metadata', {}).get('agent_used', 'Unknown')}")
                if result.get('fallback_used'):
                    print(f"   🔄 Fallback: TBI → MedPalm activated")
                print(f"   Time: {result['processing_time']:.2f}s")
                if result.get('paraphrased_query'):
                    print(f"   Paraphrased: '{result['paraphrased_query']}'")
                if result.get('confidence'):
                    print(f"   Confidence: {result['confidence']:.2f}")
                
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