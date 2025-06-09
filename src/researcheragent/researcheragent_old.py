from src.pandasagent.pandasagent2 import EnhancedPandasAgent, DB_SCHEMA

"""
Professional Researcher Agent for TBI Data Analysis
A specialized agent that routes researcher queries and provides data analysis capabilities
with medical information support
"""

"""
Professional Researcher Agent for TBI Data Analysis
A specialized agent that routes researcher queries and provides data analysis capabilities
with medical information support
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
import re
from collections import deque

# Import the PandasAgent
# from pandasagent import EnhancedPandasAgent, DB_SCHEMA

# Import medical and retrieval agents
try:
    from src.medpalm.medical_assistant import MedPalmAgent
except ImportError:
    MedPalmAgent = None
    
try:
    from src.retrieval.cdcretrieval import CDCTBIRetriever
except ImportError:
    CDCTBIRetriever = None

# Import guardrail agent if available
try:
    from src.guard.guardrailagent import MedicalGuardrailAgent
except ImportError:
    MedicalGuardrailAgent = None

logger = logging.getLogger(__name__)

class ProfessionalResearcherAgent:
    """
    Professional Researcher Agent that acts as a router and coordinator
    for researcher data analysis and medical information queries
    """
    
    def __init__(self, gemini_api_key: str = None, pinecone_api_key: str = None,
                 index_name: str = "us-cdc-tbi", embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_provider: str = "gemini", db_config: Dict[str, str] = None):
        """Initialize the Professional Researcher Agent with all agents"""
        
        # Load environment variables
        load_dotenv()
        
        # Store configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX2_NAME") or index_name
        self.embedding_model = os.getenv("EMBEDDING_MODEL") or embedding_model
        self.llm_provider = llm_provider
        
        self.db_config = db_config or {
            'user': os.getenv("user") or os.getenv("DB_USER"),
            'password': os.getenv("password") or os.getenv("DB_PASSWORD"),
            'host': os.getenv("host") or os.getenv("DB_HOST"),
            'port': os.getenv("port") or os.getenv("DB_PORT", "5432"),
            'dbname': os.getenv("dbname") or os.getenv("DB_NAME")
        }
        
        # Session management
        self.session_history = deque(maxlen=20)  # Larger buffer for research context
        self.last_assistant_message = ""
        self.current_researcher = None
        self.welcomed_researchers = set()
        
        # Power of Patients company information
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
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("Professional Researcher Agent initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all sub-agents"""
        try:
            # Initialize Gemini for routing and context
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
            
            # Initialize PandasAgent for data analysis
            self.pandas_agent = EnhancedPandasAgent(gemini_api_key=self.gemini_api_key)
            logger.info("Enhanced PandasAgent initialized")
            
            # Important: Connect to database after initialization
            # This will be done in the connect_to_database method
            
            # Initialize Medical Agent
            try:
                if MedPalmAgent and self.gemini_api_key:
                    self.medical_agent = MedPalmAgent(api_key=self.gemini_api_key)
                    logger.info("MedPalm Agent initialized")
                else:
                    self.medical_agent = None
                    logger.warning("MedPalm Agent not available")
            except Exception as e:
                logger.warning(f"MedPalm Agent initialization failed: {e}")
                self.medical_agent = None
            
            # Initialize TBI Retrieval Agent
            try:
                if CDCTBIRetriever and self.pinecone_api_key:
                    self.retrieval_agent = CDCTBIRetriever(
                        pinecone_api_key=self.pinecone_api_key,
                        index_name=self.index_name,
                        embedding_model=self.embedding_model,
                        llm_provider=self.llm_provider
                    )
                    logger.info("TBI Retrieval Agent initialized")
                else:
                    self.retrieval_agent = None
                    logger.warning("TBI Retrieval Agent not available")
            except Exception as e:
                logger.warning(f"TBI Retrieval Agent initialization failed: {e}")
                self.retrieval_agent = None
            
            # Initialize Guardrail Agent if available
            try:
                if MedicalGuardrailAgent:
                    self.guardrail_agent = MedicalGuardrailAgent(self.gemini_api_key)
                    logger.info("Medical Guardrail Agent initialized")
                else:
                    self.guardrail_agent = None
            except Exception as e:
                logger.warning(f"Guardrail Agent not available: {e}")
                self.guardrail_agent = None
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
    
    async def connect_to_database(self) -> bool:
        """Connect PandasAgent to database"""
        try:
            # Connect the PandasAgent to database
            success = self.pandas_agent.connect_to_database(self.db_config)
            
            if success:
                logger.info("Successfully connected PandasAgent to database")
                # Get schema info for context
                self.schema_info = self.pandas_agent.get_comprehensive_schema_info()
            
            return success
            
        except Exception as e:
            logger.exception(f"Failed to connect to database: {str(e)}")
            return False
    
    def _get_time_based_greeting(self) -> str:
        """Generate time-based greeting"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "Good Morning"
        elif 12 <= current_hour < 17:
            return "Good Afternoon"
        elif 17 <= current_hour < 22:
            return "Good Evening"
        else:
            return "Hello"
    
    def _generate_researcher_welcome(self, researcher_name: str = "Researcher") -> str:
        """Generate welcome message for researchers"""
        time_greeting = self._get_time_based_greeting()
        
        # Get database statistics if available
        db_stats = ""
        if hasattr(self, 'schema_info') and self.schema_info:
            # Parse schema info for basic stats
            lines = self.schema_info.split('\n')
            for line in lines:
                if "patients" in line and "Rows:" in line:
                    db_stats = f"\n\nOur database currently contains {line.split('Rows:')[1].strip().split()[0]} patient records with comprehensive TBI incident data, symptoms, therapies, and social determinants."
                    break
        
        welcome_message = f"""{time_greeting}, {researcher_name}!

I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. I'm here to help you with TBI data analysis, medical questions, TBI information, and research guidance.{db_stats}

I can assist with:
• **Data Analysis**: Statistical analysis, demographics, treatment outcomes, visualizations
• **Medical Information**: General medical questions, conditions, treatments
• **TBI Knowledge**: Specific information about traumatic brain injury, concussions, symptoms
• **Research Methodology**: Statistical approaches, study design guidance

How can I assist you with your research analysis today?"""
        
        return welcome_message
    
    def _format_recent_history(self, num_entries: int = 5) -> str:
        """Format recent conversation history"""
        history_parts = []
        for exchange in list(self.session_history)[-num_entries:]:
            history_parts.append(f"Researcher: {exchange['query']}")
            # Truncate long responses but keep key info
            response = exchange['response']
            if len(response) > 200:
                response = response[:200] + "..."
            history_parts.append(f"Assistant: {response}")
        return "\n".join(history_parts)
    
    def _update_conversation_memory(self, query: str, response: str, metadata: Dict = None):
        """Update conversation memory with exchange"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "researcher": self.current_researcher,
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        self.session_history.append(exchange)
        self.last_assistant_message = response
    
    async def _classify_research_query(self, query: str) -> str:
        """Classify research query intent with medical query support"""
        try:
            recent_history = self._format_recent_history(num_entries=3) if self.session_history else ""
            
            classification_prompt = f"""
You are classifying a researcher's query. Researchers may ask about data analysis, medical information, or TBI knowledge.

{"RECENT CONVERSATION:" if recent_history else ""}
{recent_history}

RESEARCHER QUERY: "{query}"

CLASSIFICATION OPTIONS:
1. "data_analysis" - Queries requiring statistical analysis, data exploration, or visualization of the TBI database
   Examples: "What are the most common TBI causes?", "Show age distribution", "Correlation between X and Y"
   
2. "medical_general" - General medical questions NOT requiring database analysis
   Examples: "What is diabetes?", "How do beta blockers work?", "Treatment for hypertension"
   
3. "tbi_knowledge" - Questions about TBI, concussion, or brain injury knowledge (not database queries)
   Examples: "What are TBI symptoms?", "How is concussion diagnosed?", "TBI recovery timeline"
   
4. "schema_info" - Questions about database structure, tables, columns, or data organization
   Examples: "What tables are available?", "How many tables do we have?", "What columns are in patients table?", "Show me the database schema"
   
5. "methodology" - Questions about research methods, statistical approaches, or analysis guidance
   Examples: "What statistical test to use?", "How to analyze longitudinal data?"
   
6. "general_conversation" - Greetings, thanks, identity questions, or questions about Power of Patients/Sallie
   Examples: "Hello", "Who are you?", "What is Power of Patients?", "Who is Sallie?"

IMPORTANT: 
- Questions about "how many tables", "what tables", "database structure" should be "schema_info"
- Questions about "Power of Patients", "Sallie", or identity ("who are you") should be "general_conversation"
- If asking for analysis of specific data in tables, classify as "data_analysis"
- If asking for general medical/TBI knowledge without data analysis, classify as "medical_general" or "tbi_knowledge"

Respond with ONLY one option.

Classification:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    classification_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            classification = response.text.strip().lower()
            
            # Validate and extract
            valid_intents = ["data_analysis", "medical_general", "tbi_knowledge", "schema_info", 
                           "methodology", "general_conversation"]
            
            for intent in valid_intents:
                if intent in classification:
                    logger.info(f"Research query classified as: {intent}")
                    return intent
            
            # Fallback classification based on keywords
            query_lower = query.lower()
            
            # Check for Power of Patients, Sallie, or identity questions first
            identity_keywords = ['power of patients', 'sallie', 'who are you', 'who is sallie', 
                               'what is power of patients', 'tell me about power of patients']
            if any(keyword in query_lower for keyword in identity_keywords):
                return "general_conversation"
            
            # Schema keywords - check these BEFORE data analysis
            # Special handling for "how many tables" which could be confused with data analysis
            if 'table' in query_lower or 'schema' in query_lower or 'column' in query_lower:
                # These are schema questions, not data analysis
                schema_phrases = ['how many table', 'what table', 'list table', 'show table',
                                'database schema', 'database structure', 'table structure']
                if any(phrase in query_lower for phrase in schema_phrases):
                    return "schema_info"
                # General table/column questions
                if any(word in query_lower for word in ['table', 'tables', 'column', 'columns', 'schema']):
                    return "schema_info"
            
            # Schema keywords - check these BEFORE data analysis
            schema_keywords = ['table', 'tables', 'column', 'columns', 'schema', 'database structure', 
                             'field', 'fields', 'how many table', 'what table', 'list table']
            if any(keyword in query_lower for keyword in schema_keywords):
                return "schema_info"
            
            # Data analysis keywords
            analysis_keywords = ['show', 'analyze', 'compare', 'distribution', 'correlation', 
                               'most common', 'average', 'trend', 'visualize', 'chart', 
                               'graph', 'plot', 'percentage', 'statistics']
            # Only classify as data_analysis if it's not about schema
            if any(keyword in query_lower for keyword in analysis_keywords) and not any(schema in query_lower for schema in ['table', 'column', 'schema']):
                return "data_analysis"
            
            # TBI knowledge keywords (not data analysis)
            tbi_keywords = ['tbi', 'traumatic brain', 'concussion', 'brain injury', 'head trauma']
            medical_context = ['what is', 'symptoms', 'treatment', 'diagnosis', 'recovery']
            if any(tbi in query_lower for tbi in tbi_keywords) and any(context in query_lower for context in medical_context):
                return "tbi_knowledge"
            
            # General medical keywords
            medical_keywords = ['diabetes', 'heart', 'blood pressure', 'medication', 'disease', 
                              'condition', 'treatment', 'symptoms', 'diagnosis']
            if any(keyword in query_lower for keyword in medical_keywords):
                return "medical_general"
            
            # Schema keywords
            schema_keywords = ['table', 'column', 'schema', 'database structure', 'field']
            if any(keyword in query_lower for keyword in schema_keywords):
                return "schema_info"
            
            return "general_conversation"
            
        except Exception as e:
            logger.error(f"Error in query classification: {e}")
            # Emergency fallback
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['power of patients', 'sallie', 'who are you']):
                return "general_conversation"
            return "general_conversation"
    
    async def _handle_data_analysis(self, query: str) -> Dict[str, Any]:
        """Handle data analysis queries using PandasAgent"""
        try:
            # Call PandasAgent to process the query
            result = await self.pandas_agent.process_query(query)
            
            # Debug logging
            logger.info(f"PandasAgent result keys: {result.keys()}")
            logger.info(f"Visualization exists: {bool(result.get('visualization'))}")
            
            # Extract components
            answer = result.get('answer', 'No results found.')
            visualization_html = result.get('visualization', None)
            metadata = result.get('metadata', {})
            
            # Log visualization info
            if visualization_html:
                logger.info(f"Visualization HTML length: {len(visualization_html)}")
            else:
                logger.warning("No visualization HTML returned from PandasAgent")
            
            # Format the response with embedded visualization
            if visualization_html:
                # Create a combined response with text and visualization
                formatted_response = f"""{answer}

📊 **Visualization:**
<div class="visualization-container">
{visualization_html}
</div>"""
                
                # Also return visualization separately for flexible rendering
                return {
                    "response": answer,  # Text-only version
                    "visualization_html": visualization_html,
                    "combined_response": formatted_response,
                    "has_visualization": True,
                    "metadata": metadata
                }
            else:
                return {
                    "response": answer,
                    "visualization_html": None,
                    "combined_response": answer,
                    "has_visualization": False,
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return {
                "response": f"I encountered an error analyzing the data: {str(e)}",
                "visualization_html": None,
                "combined_response": f"I encountered an error analyzing the data: {str(e)}",
                "has_visualization": False,
                "metadata": {"error": str(e)}
            }
    
    async def _handle_medical_query(self, query: str) -> str:
        """Handle general medical queries using MedPalm agent"""
        try:
            if self.medical_agent:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.medical_agent.process_query(query)
                )
                
                # Log response type for debugging
                logger.info(f"MedPalm response type: {type(response)}")
                if isinstance(response, dict):
                    logger.info(f"MedPalm response keys: {response.keys()}")
                
                # Handle different response types
                if isinstance(response, dict):
                    # If MedPalmAgent returns a dict, extract the text response
                    text_response = response.get('response', response.get('answer', response.get('text', '')))
                    if not text_response:
                        # If no standard keys, convert whole dict to string
                        text_response = str(response)
                    return text_response
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            else:
                return "Medical information agent is not available. For medical questions, please consult medical literature or healthcare professionals."
                
        except Exception as e:
            logger.error(f"Error calling medical agent: {e}")
            return "I encountered an error accessing medical information. Please try rephrasing your question."
    
    async def _handle_tbi_knowledge(self, query: str) -> str:
        """Handle TBI-specific knowledge queries using retrieval agent"""
        try:
            if self.retrieval_agent:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self.retrieval_agent.ask_question(query, top_k=8)
                )
                
                # Handle different response types
                if isinstance(result, dict):
                    return result.get('answer', result.get('response', str(result)))
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)
            else:
                # Fallback to general medical agent if available
                if self.medical_agent:
                    return await self._handle_medical_query(query)
                else:
                    return "TBI information system is not available. Please consult medical literature for TBI-specific information."
                
        except Exception as e:
            logger.error(f"Error calling TBI retrieval agent: {e}")
            return "I encountered an error accessing TBI information. Please try rephrasing your question."
    
    async def _handle_schema_info(self, query: str) -> str:
        """Handle schema information queries"""
        try:
            # Quick responses for common schema questions
            query_lower = query.lower()
            
            if 'how many table' in query_lower:
                # Count tables from schema
                table_count = 7  # Based on DB_SCHEMA: patients, tbi_incidents, symptom_logs, worst_symptoms, therapies, social_determinants, symptom_reference
                return f"""We have **7 main tables** in our TBI database:

1. **patients** - Core patient demographics and information
2. **tbi_incidents** - TBI incident details and causes  
3. **symptom_logs** - Longitudinal symptom tracking
4. **worst_symptoms** - Patient-reported worst symptoms
5. **therapies** - Treatment and therapy information
6. **social_determinants** - Social factors affecting health
7. **symptom_reference** - Reference data for symptoms

Each table contains specific columns for comprehensive TBI research. Would you like details about any specific table?"""
            
            # For other schema queries, use the full prompt
            schema_prompt = f"""
You are helping a researcher understand the TBI database schema.

AVAILABLE SCHEMA INFORMATION:
{DB_SCHEMA}

DETAILED SCHEMA ANALYSIS:
{self.schema_info if hasattr(self, 'schema_info') else 'Schema analysis not available'}

RESEARCHER QUESTION: "{query}"

Provide a clear, detailed response about the database schema, focusing on:
1. Relevant tables and their purpose
2. Key columns and data types
3. Relationships between tables
4. Data quality considerations
5. Suggestions for analysis approaches

Keep the response informative but concise.

Response:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.gemini_model.generate_content(
                    schema_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error handling schema query: {e}")
            return f"Here's the basic schema information:\n\n{DB_SCHEMA}"
    
    async def _handle_methodology(self, query: str) -> str:
        """Handle research methodology queries"""
        try:
            methodology_prompt = f"""
You are a research methodology advisor for TBI data analysis.

RESEARCHER QUESTION: "{query}"

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
                    safety_settings=self.safety_settings
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error handling methodology query: {e}")
            return "I encountered an error providing methodology guidance. For statistical analysis of TBI data, consider consulting with a biostatistician."
    
    def _get_medical_disclaimer(self, intent: str) -> str:
        """Get appropriate medical disclaimer for researchers"""
        if intent in ["medical_general", "tbi_knowledge"]:
            return "\n\n📚 **Research Note:** This information is provided for research and educational purposes. For clinical applications or patient care decisions, always refer to peer-reviewed literature and clinical guidelines."
        return ""
    
    async def process_query(self, query: str, researcher_id: str = "default") -> Dict[str, Any]:
        """Main method to process researcher queries"""
        start_time = datetime.now()
        
        try:
            # Set current researcher
            self.current_researcher = researcher_id
            
            # Initialize welcome flags
            show_welcome = False
            welcome_prefix = ""
            
            # Check if this is a new researcher
            if researcher_id not in self.welcomed_researchers:
                self.welcomed_researchers.add(researcher_id)
                show_welcome = True
                # Generate welcome but don't return immediately
                welcome_prefix = f"{self._generate_researcher_welcome(researcher_id)}\n\n---\n\nNow, let me help you with your query: \"{query}\"\n\n"
            
            # Check guardrail if available (less strict for researchers)
            if self.guardrail_agent:
                try:
                    # Skip guardrail for legitimate research questions
                    research_keywords = [
                        # Identity/company
                        'who are you', 'power of patients', 'sallie', 'tell me about yourself',
                        # Schema/database questions
                        'table', 'tables', 'column', 'columns', 'schema', 'database', 'data',
                        # Analysis questions
                        'how many', 'show me', 'analyze', 'distribution', 'correlation',
                        # Medical/TBI questions
                        'tbi', 'traumatic brain', 'concussion', 'symptom', 'treatment'
                    ]
                    
                    skip_guardrail = any(keyword in query.lower() for keyword in research_keywords)
                    
                    if not skip_guardrail:
                        guardrail_result = await self.guardrail_agent.process_query(query)
                        
                        # Only block clearly inappropriate content for researchers with very high confidence
                        if not guardrail_result['allow'] and guardrail_result['classification'].get('confidence', 0) > 0.98:
                            logger.warning(f"Guardrail blocked query: {query} with confidence {guardrail_result['classification'].get('confidence', 0)}")
                            return {
                                "success": True,
                                "researcher_id": researcher_id,
                                "query": query,
                                "intent": "blocked",
                                "response": "This query appears to be outside the scope of research analysis. Please focus on research-related questions.",
                                "visualization_html": None,
                                "combined_response": "This query appears to be outside the scope of research analysis.",
                                "has_visualization": False,
                                "processing_time": (datetime.now() - start_time).total_seconds()
                            }
                    else:
                        logger.info(f"Skipped guardrail for research query: {query}")
                except Exception as e:
                    logger.warning(f"Guardrail check failed: {e}")
            
            # Classify query intent
            intent = await self._classify_research_query(query)
            logger.info(f"Query '{query}' classified as: {intent}")
            
            # Route to appropriate handler
            visualization_html = None
            has_visualization = False
            metadata = {}
            
            if intent == "data_analysis":
                analysis_result = await self._handle_data_analysis(query)
                response = analysis_result["response"]
                visualization_html = analysis_result["visualization_html"]
                combined_response = analysis_result["combined_response"]
                has_visualization = analysis_result["has_visualization"]
                metadata = analysis_result["metadata"]
                
            elif intent == "medical_general":
                response = await self._handle_medical_query(query)
                # Ensure response is a string before adding disclaimer
                response = str(response) if response else "Unable to process medical query."
                # Add medical disclaimer for researchers
                response = response + self._get_medical_disclaimer(intent)
                visualization_html = None
                combined_response = response
                has_visualization = False
                metadata = {"agent_used": "MedPalm"}
                
            elif intent == "tbi_knowledge":
                response = await self._handle_tbi_knowledge(query)
                # Ensure response is a string before adding disclaimer
                response = str(response) if response else "Unable to process TBI query."
                # Add medical disclaimer for researchers
                response = response + self._get_medical_disclaimer(intent)
                visualization_html = None
                combined_response = response
                has_visualization = False
                metadata = {"agent_used": "TBI Retrieval"}
                
            elif intent == "schema_info":
                response = await self._handle_schema_info(query)
                visualization_html = None
                combined_response = response
                has_visualization = False
                metadata = {}
                
            elif intent == "methodology":
                response = await self._handle_methodology(query)
                visualization_html = None
                combined_response = response
                has_visualization = False
                metadata = {}
                
            else:  # general_conversation
                # Handle identity and company questions
                if any(keyword in query.lower() for keyword in ['power of patients', 'what is power of patients']):
                    response = self.company_info
                elif any(keyword in query.lower() for keyword in ['who are you', 'who is sallie', 'tell me about yourself']):
                    response = f"""I'm Sallie, a professional and empathetic healthcare assistant created by Power of Patients. 

I'm designed to help researchers like you with:
• TBI data analysis and visualization
• Medical information queries
• Research methodology guidance
• Understanding our comprehensive TBI database

{self.company_info}

How can I assist you with your research today?"""
                else:
                    response = f"I understand you're saying: {query}. I'm Sallie, your research assistant from Power of Patients. I'm here to help with TBI research, data analysis, and medical information. What would you like to explore?"
                
                visualization_html = None
                combined_response = response
                has_visualization = False
                metadata = {}
            
            # Update conversation memory
            self._update_conversation_memory(query, response, metadata)
            
            # Add welcome prefix if this is a new researcher
            if show_welcome:
                response = welcome_prefix + response
                combined_response = welcome_prefix + combined_response
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "researcher_id": researcher_id,
                "query": query,
                "intent": intent,
                "response": response,  # Text-only response
                "visualization_html": visualization_html,  # Separate visualization
                "combined_response": combined_response,  # Combined text + viz
                "has_visualization": has_visualization,
                "processing_time": processing_time,
                "metadata": metadata,
                "medical_disclaimer_added": intent in ["medical_general", "tbi_knowledge"],
                "welcome_shown": show_welcome
            }
            
        except Exception as e:
            logger.exception(f"Error processing query: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again.",
                "visualization_html": None,
                "combined_response": "I encountered an error processing your request. Please try again.",
                "has_visualization": False,
                "processing_time": processing_time
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current research session"""
        # Extract unique analysis types from session
        analysis_types = set()
        visualizations_created = 0
        agents_used = set()
        
        for exchange in self.session_history:
            metadata = exchange.get('metadata', {})
            if metadata.get('chart_type'):
                visualizations_created += 1
            if metadata.get('agent_used'):
                agents_used.add(metadata['agent_used'])
            if 'intent' in metadata:
                analysis_types.add(metadata['intent'])
        
        # Count different query types
        query_intents = {}
        for exchange in self.session_history:
            intent = exchange.get('metadata', {}).get('intent', 'unknown')
            query_intents[intent] = query_intents.get(intent, 0) + 1
        
        return {
            "researcher_id": self.current_researcher,
            "session_length": len(self.session_history),
            "analyses_performed": len([e for e in self.session_history if e.get('metadata', {}).get('intent') == 'data_analysis']),
            "visualizations_created": visualizations_created,
            "medical_queries": query_intents.get('medical_general', 0) + query_intents.get('tbi_knowledge', 0),
            "agents_used": list(agents_used),
            "query_breakdown": query_intents,
            "total_processing_time": sum([e.get('processing_time', 0) for e in self.session_history if 'processing_time' in e]),
            "database_connected": hasattr(self.pandas_agent, 'dataframes') and bool(self.pandas_agent.dataframes),
            "available_agents": {
                "pandas_agent": True,
                "medical_agent": self.medical_agent is not None,
                "retrieval_agent": self.retrieval_agent is not None,
                "guardrail_agent": self.guardrail_agent is not None
            }
        }
    
    def clear_session(self):
        """Clear current session"""
        self.session_history.clear()
        self.last_assistant_message = ""
        logger.info("Research session cleared")
    
    def get_available_analyses(self) -> Dict[str, List[str]]:
        """Get categorized list of available analyses and queries"""
        return {
            "data_analysis": [
                "What are the most common causes of TBI in our patient population?",
                "Show the age distribution of TBI patients by gender",
                "What are the most frequent immediate symptoms after TBI?",
                "Which states have the highest number of TBI patients?",
                "What's the relationship between injury location and symptom severity?",
                "What therapies are most commonly prescribed for TBI patients?",
                "How does veteran status relate to TBI causes?",
                "What percentage of patients had TBI before their current incident?",
                "Show symptom categories by frequency and average severity",
                "What are the top social determinants affecting TBI patients?"
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
            ]
        }


# Terminal Testing Interface
async def terminal_interface():
    """Interactive terminal interface for testing ResearcherAgent"""
    print("=" * 60)
    print("TBI RESEARCHER AGENT - TERMINAL INTERFACE")
    print("=" * 60)
    
    # Initialize agent
    print("Initializing Researcher Agent...")
    agent = ProfessionalResearcherAgent()
    
    # Connect to database
    print("Connecting to TBI database...")
    if not await agent.connect_to_database():
        print("❌ Failed to connect to database. Please check your configuration.")
        return
    
    print("✅ Researcher Agent initialized successfully!")
    
    # Show available agents
    summary = agent.get_session_summary()
    print("\n🔧 Available Agents:")
    for agent_name, available in summary['available_agents'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {agent_name}")
    
    print("\nCommands:")
    print("- Enter your query normally (data analysis, medical, or TBI questions)")
    print("- Type 'examples' to see example queries by category")
    print("- Type 'schema' to see database schema")
    print("- Type 'summary' to see session summary")
    print("- Type 'clear' to clear session")
    print("- Type 'quit' to exit")
    print("-" * 60)
    
    researcher_id = input("\nEnter your researcher ID (or press Enter for 'default'): ").strip() or "default"
    
    while True:
        try:
            user_input = input(f"\n[Researcher {researcher_id}] Query: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                agent.clear_session()
                print("✅ Session cleared")
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_session_summary()
                print("\n📊 SESSION SUMMARY:")
                print(json.dumps(summary, indent=2))
                continue
            elif user_input.lower() == 'examples':
                print("\n📝 EXAMPLE QUERIES BY CATEGORY:")
                examples = agent.get_available_analyses()
                for category, queries in examples.items():
                    print(f"\n{category.upper().replace('_', ' ')}:")
                    for i, example in enumerate(queries[:5], 1):  # Show first 5 of each
                        print(f"  {i}. {example}")
                continue
            elif user_input.lower() == 'schema':
                print("\n📋 DATABASE SCHEMA:")
                print(agent.pandas_agent.get_comprehensive_schema_info() if hasattr(agent.pandas_agent, 'get_comprehensive_schema_info') else DB_SCHEMA)
                continue
            
            # Process query
            print("\n⏳ Processing query...")
            result = await agent.process_query(user_input, researcher_id)
            
            if result["success"]:
                # Display response
                print(f"\n📊 RESPONSE:")
                print(result['response'])
                
                # If there's a visualization, indicate it clearly
                if result.get('has_visualization') and result.get('visualization_html'):
                    print("\n📈 VISUALIZATION GENERATED:")
                    print("(In a web interface, the interactive chart would appear here)")
                    print(f"Visualization Type: {result.get('metadata', {}).get('chart_type', 'Unknown')}")
                    print(f"HTML Length: {len(result['visualization_html'])} characters")
                    
                    # Optionally save to file for viewing
                    save_viz = input("\nSave visualization to HTML file? (y/n): ").strip().lower()
                    if save_viz == 'y':
                        filename = f"research_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(result['visualization_html'])
                        print(f"✅ Visualization saved to: {filename}")
                elif result.get('has_visualization'):
                    print("\n⚠️ Visualization was expected but HTML not generated")
                    print(f"Debug info: has_visualization={result.get('has_visualization')}, html_exists={bool(result.get('visualization_html'))}")
                
                # Show metadata
                intent = result['intent']
                agent_used = result.get('metadata', {}).get('agent_used', 'Default')
                disclaimer = "📚 Medical disclaimer added" if result.get('medical_disclaimer_added') else ""
                
                print(f"\n📍 Intent: {intent} | Agent: {agent_used} | Time: {result['processing_time']:.2f}s {disclaimer}")
            else:
                print(f"\n❌ Error: {result['error']}")
                print(f"Response: {result['response']}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run terminal interface
    asyncio.run(terminal_interface())