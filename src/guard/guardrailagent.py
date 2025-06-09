import logging
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, Optional, Any, List
from datetime import datetime
import re
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class MedicalGuardrailAgent:
    """
    Guardrail Agent that detects non-medical queries and redirects patients
    to appropriate medical/TBI-related topics
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the Medical Guardrail Agent"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.model = None
        self.safety_settings = None
        
        # Predefined redirect responses
        self.redirect_responses = {
            "general": """I'm Sallie, your healthcare assistant specializing in medical and TBI-related questions. 

I'm here to help you with:
• Medical conditions and symptoms
• Traumatic Brain Injury (TBI) and concussion information
• Treatment options and recovery guidance
• Healthcare resources and support

How can I assist you with your health or TBI-related concerns today?""",
            
            "personal": """I'm Sallie, your dedicated healthcare assistant. While I'd love to chat about everything, I'm specifically designed to help you with medical and TBI-related questions.

I can assist you with:
• Understanding your symptoms
• TBI and concussion recovery
• Medical treatment information  
• Healthcare guidance and resources

What health-related questions can I help you with?""",
            
            "technology": """I'm Sallie, your healthcare assistant. I focus on medical and TBI-related topics rather than technology questions.

I'm here to help you with:
• Medical conditions and treatments
• Brain injury and concussion support
• Symptom management
• Healthcare resources

Is there anything about your health or recovery that I can help you with?""",
            
            "entertainment": """I'm Sallie, your healthcare assistant. While entertainment is great, I specialize in medical and TBI-related support.

I can help you with:
• Medical questions and concerns
• TBI and concussion information
• Recovery and treatment guidance
• Healthcare resources

What medical or health-related questions do you have for me?""",
            
            "general_knowledge": """I'm Sallie, your healthcare assistant. I focus on medical and TBI-related information rather than general knowledge questions.

I'm here to support you with:
• Medical conditions and symptoms
• Brain injury and concussion care
• Treatment and recovery guidance
• Healthcare resources and support

How can I help you with your health or TBI-related concerns?"""
        }
        
        self._initialize_model()
        logger.info("Medical Guardrail Agent initialized successfully")
    
    def _initialize_model(self):
        """Initialize the Gemini model for guardrail detection"""
        try:
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                
                self.safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini model initialized for guardrail detection")
            else:
                logger.error("No Gemini API key provided for guardrail agent")
                
        except Exception as e:
            logger.error(f"Error initializing guardrail model: {e}")
    
    async def check_query_appropriateness(self, query: str) -> Dict[str, Any]:
        """
        Check if the query is appropriate for a medical/TBI healthcare assistant
        
        Returns:
            Dict with 'is_medical_related', 'category', 'confidence', and 'reasoning'
        """
        try:
            # Quick keyword-based pre-screening for efficiency
            quick_result = self._quick_keyword_screening(query)
            if quick_result['high_confidence']:
                return quick_result
            
            # Use LLM for more nuanced detection
            llm_result = await self._llm_based_screening(query)
            return llm_result
            
        except Exception as e:
            logger.error(f"Error in guardrail check: {e}")
            # Default to allowing the query if there's an error
            return {
                'is_medical_related': True,
                'category': 'medical_general',
                'confidence': 0.5,
                'reasoning': f'Error in screening: {str(e)}',
                'redirect_needed': False
            }
    
    def _quick_keyword_screening(self, query: str) -> Dict[str, Any]:
        """Quick keyword-based screening for obvious non-medical queries"""
        query_lower = query.lower().strip()
        
        # Sexual/Inappropriate content keywords (BLOCK with high priority)
        sexual_inappropriate_keywords = [
            'sex', 'sexual', 'intercourse', 'orgasm', 'masturbation', 'masturbate',
            'porn', 'pornography', 'erotic', 'arousal', 'climax', 'ejaculation',
            'foreplay', 'intimate', 'intimacy', 'libido', 'viagra', 'cialis',
            'condom', 'contraception', 'birth control', 'pregnancy test',
            'std', 'sti', 'sexually transmitted', 'genital', 'penis', 'vagina',
            'breast', 'nipple', 'clitoris', 'vulva', 'anus', 'anal',
            'blow job', 'oral sex', 'hookup', 'dating', 'relationship',
            'horny', 'sexy', 'nude', 'naked', 'strip', 'seduce'
        ]
        
        # Medical/Health related keywords (ALLOW)
        medical_keywords = [
            # Symptoms
            'pain', 'headache', 'dizzy', 'nausea', 'fatigue', 'tired', 'sleep', 
            'memory', 'confusion', 'balance', 'vision', 'hearing', 'mood',
            'anxiety', 'depression', 'irritable', 'concentration', 'focus',
            
            # Medical terms
            'doctor', 'hospital', 'medication', 'treatment', 'therapy', 'symptom',
            'diagnosis', 'medical', 'health', 'healthcare', 'clinic', 'prescription',
            'surgery', 'recovery', 'healing', 'injury', 'accident', 'trauma',
            
            # TBI specific
            'tbi', 'concussion', 'brain injury', 'head injury', 'brain trauma',
            'post-concussion', 'traumatic brain', 'cognitive', 'neurological',
            
            # Body parts (medical context)
            'head', 'brain', 'neck', 'back', 'chest', 'stomach', 'heart', 'lung',
            
            # Health conditions
            'diabetes', 'hypertension', 'blood pressure', 'cholesterol', 'fever',
            'infection', 'virus', 'bacteria', 'disease', 'condition', 'syndrome',
            
            # Exercise/fitness (legitimate medical context)
            'exercise', 'workout', 'fitness', 'muscle', 'strength', 'cardio'
        ]
        
        # Obviously non-medical keywords (BLOCK)
        non_medical_keywords = [
            # Technology
            'computer', 'software', 'programming', 'coding', 'website', 'app',
            'internet', 'wifi', 'bluetooth', 'iphone', 'android', 'windows', 'mac',
            
            # Entertainment
            'movie', 'music', 'song', 'game', 'gaming', 'netflix', 'youtube',
            'celebrity', 'actor', 'actress', 'singer', 'band', 'album',
            
            # Politics/People
            'president', 'politics', 'government', 'election', 'obama', 'trump',
            'biden', 'politician', 'congress', 'senate',
            
            # General knowledge
            'history', 'geography', 'science', 'math', 'physics', 'chemistry',
            'biology', 'astronomy', 'space', 'planet', 'universe',
            
            # Sports
            'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
            'olympics', 'team', 'player', 'coach', 'score',
            
            # Business/Finance (non-medical)
            'stock', 'investment', 'trading', 'cryptocurrency', 'bitcoin',
            'business', 'marketing', 'sales', 'profit',
            
            # Transportation
            'car', 'vehicle', 'truck', 'motorcycle', 'airplane', 'train',
            'bus', 'driving', 'parking', 'traffic',
            
            # Food (non-medical context)
            'recipe', 'cooking', 'restaurant', 'menu', 'chef'
        ]
        
        # Check for sexual/inappropriate content first (highest priority)
        sexual_matches = sum(1 for keyword in sexual_inappropriate_keywords if keyword in query_lower)
        if sexual_matches >= 1:
            # Determine if it's sexual health vs inappropriate
            medical_context = any(word in query_lower for word in ['health', 'doctor', 'medical', 'treatment', 'disease', 'infection', 'problem'])
            
            if medical_context and sexual_matches == 1:
                # Might be legitimate sexual health question
                category = 'sexual_health'
            else:
                # Likely inappropriate content
                category = 'inappropriate_content'
                
            return {
                'is_medical_related': False,
                'category': category,
                'confidence': 0.95,
                'reasoning': f'Sexual/inappropriate content detected: {sexual_matches} matches',
                'high_confidence': True,
                'redirect_needed': True
            }
        
        # Count other matches
        medical_matches = sum(1 for keyword in medical_keywords if keyword in query_lower)
        non_medical_matches = sum(1 for keyword in non_medical_keywords if keyword in query_lower)
        
        # High confidence decisions for medical content
        if medical_matches >= 2 and non_medical_matches == 0:
            return {
                'is_medical_related': True,
                'category': 'medical_general',
                'confidence': 0.9,
                'reasoning': f'Multiple medical keywords detected: {medical_matches} matches',
                'high_confidence': True,
                'redirect_needed': False
            }
        
        # High confidence decisions for non-medical content
        if non_medical_matches >= 1 and medical_matches == 0:
            category = self._categorize_non_medical_query(query_lower)
            return {
                'is_medical_related': False,
                'category': category,
                'confidence': 0.9,
                'reasoning': f'Non-medical keywords detected: {non_medical_matches} matches',
                'high_confidence': True,
                'redirect_needed': True
            }
        
        # Uncertain cases - let LLM decide
        return {
            'high_confidence': False,
            'medical_matches': medical_matches,
            'non_medical_matches': non_medical_matches,
            'sexual_matches': sexual_matches
        }
    
    def _categorize_non_medical_query(self, query_lower: str) -> str:
        """Categorize non-medical queries for appropriate redirect response"""
        # Sexual/inappropriate content gets highest priority
        if any(word in query_lower for word in ['sex', 'sexual', 'intimate', 'dating', 'relationship', 'porn', 'erotic']):
            # Check if it might be legitimate sexual health
            if any(word in query_lower for word in ['health', 'doctor', 'medical', 'treatment', 'disease']):
                return 'sexual_health'
            else:
                return 'inappropriate_content'
        elif any(word in query_lower for word in ['computer', 'software', 'app', 'tech', 'programming']):
            return 'technology'
        elif any(word in query_lower for word in ['movie', 'music', 'game', 'entertainment', 'celebrity']):
            return 'entertainment'
        elif any(word in query_lower for word in ['obama', 'trump', 'president', 'politics', 'government']):
            return 'personal'
        elif any(word in query_lower for word in ['history', 'geography', 'science', 'math', 'space']):
            return 'general_knowledge'
        else:
            return 'general'
    
    async def _llm_based_screening(self, query: str) -> Dict[str, Any]:
        """Use LLM for nuanced medical vs non-medical classification"""
        try:
            screening_prompt = f"""
You are a strict medical content filter for a professional healthcare assistant named Sallie. Your job is to determine if a patient's query is appropriate for a professional medical/TBI healthcare setting.

USER QUERY: "{query}"

MEDICAL/HEALTH TOPICS (ALLOW):
- Medical symptoms, conditions, treatments
- Mental health and wellness (professional context)
- Medications and prescriptions  
- Healthcare providers and services
- Injuries, accidents, and recovery
- TBI, concussion, brain injury topics
- Body functions and health concerns (medical context)
- Preventive care and health maintenance
- Emergency medical situations

INAPPROPRIATE/NON-MEDICAL TOPICS (BLOCK):
- Sexual content, intimate relationships, dating advice
- Technology, computers, programming, social media
- Entertainment, movies, music, games, celebrities
- Politics, government, politicians (unless health policy)
- General knowledge, history, geography, education
- Sports scores and statistics (non-medical)
- Business, finance, investments, shopping
- Transportation, cars, vehicles (unless TBI-related)
- Cooking recipes, food (unless medical nutrition)
- Personal information about public figures
- Inappropriate or suggestive content

STRICT GUIDELINES:
- Sexual health questions should be redirected to specialized providers
- Exercise questions mixed with sexual content = BLOCK
- Any inappropriate or suggestive content = BLOCK
- When in doubt about appropriateness, BLOCK
- Professional medical setting standards apply

EXAMPLES:
- "Can I build muscle if I have sex?" = NON-MEDICAL (inappropriate fitness/sexual content)
- "Muscle building after TBI" = MEDICAL (legitimate TBI recovery)
- "Sexual side effects of medication" = NON-MEDICAL (redirect to sexual health specialist)
- "Headache after injury" = MEDICAL (legitimate symptom)

Respond with:
CLASSIFICATION: [MEDICAL or NON-MEDICAL]
CONFIDENCE: [0.1-1.0] 
CATEGORY: [medical_general/tbi_related/inappropriate_content/sexual_health/etc.]
REASONING: [Brief explanation focusing on professional healthcare context]

Classification:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    screening_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            result = self._parse_llm_response(response.text, query)
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM screening: {e}")
            # Default to blocking potentially inappropriate queries if LLM fails
            query_lower = query.lower()
            sexual_indicators = ['sex', 'sexual', 'intimate', 'dating', 'relationship']
            if any(indicator in query_lower for indicator in sexual_indicators):
                return {
                    'is_medical_related': False,
                    'category': 'inappropriate_content',
                    'confidence': 0.8,
                    'reasoning': f'LLM error, blocking potentially inappropriate content: {str(e)}',
                    'redirect_needed': True
                }
            
            # Default to allowing other queries if LLM fails
            return {
                'is_medical_related': True,
                'category': 'medical_general', 
                'confidence': 0.5,
                'reasoning': f'LLM error, defaulting to allow: {str(e)}',
                'redirect_needed': False
            }
    
    def _parse_llm_response(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """Parse the LLM response for guardrail decision"""
        try:
            lines = response_text.strip().split('\n')
            result = {
                'original_query': original_query,
                'redirect_needed': False
            }
            
            for line in lines:
                if line.startswith('CLASSIFICATION:'):
                    classification = line.replace('CLASSIFICATION:', '').strip().upper()
                    result['is_medical_related'] = classification == 'MEDICAL'
                    result['redirect_needed'] = classification == 'NON-MEDICAL'
                    
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                        result['confidence'] = max(0.1, min(1.0, confidence))
                    except:
                        result['confidence'] = 0.7
                        
                elif line.startswith('CATEGORY:'):
                    result['category'] = line.replace('CATEGORY:', '').strip().lower()
                    
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.replace('REASONING:', '').strip()
            
            # Set defaults if parsing failed
            if 'is_medical_related' not in result:
                result['is_medical_related'] = True
                result['redirect_needed'] = False
                
            if 'confidence' not in result:
                result['confidence'] = 0.7
                
            if 'category' not in result:
                result['category'] = 'medical_general' if result['is_medical_related'] else 'general'
                
            if 'reasoning' not in result:
                result['reasoning'] = 'LLM classification completed'
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                'is_medical_related': True,
                'category': 'medical_general',
                'confidence': 0.5,
                'reasoning': f'Parse error, defaulting to allow: {str(e)}',
                'redirect_needed': False
            }
    
    def get_redirect_response(self, category: str = "general") -> str:
        """Get appropriate redirect response based on query category"""
        return self.redirect_responses.get(category, self.redirect_responses["general"])
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process a query through the guardrail
        
        Returns:
            - allow: bool - whether to allow the query to proceed
            - response: str - redirect response if not allowed
            - classification: dict - detailed classification info
        """
        try:
            # Check if query is medical-related
            classification = await self.check_query_appropriateness(query)
            
            if classification['is_medical_related']:
                return {
                    'allow': True,
                    'response': None,
                    'classification': classification,
                    'action': 'proceed_to_medical_agents'
                }
            else:
                # Generate redirect response
                redirect_response = self.get_redirect_response(classification.get('category', 'general'))
                
                return {
                    'allow': False,
                    'response': redirect_response,
                    'classification': classification,
                    'action': 'redirect_to_medical_topics'
                }
                
        except Exception as e:
            logger.error(f"Error processing guardrail query: {e}")
            # Default to allowing the query if there's an error
            return {
                'allow': True,
                'response': None,
                'classification': {
                    'error': str(e),
                    'is_medical_related': True,
                    'confidence': 0.5
                },
                'action': 'proceed_with_error'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about guardrail performance (placeholder for future implementation)"""
        return {
            'total_queries': 0,
            'medical_allowed': 0,
            'non_medical_redirected': 0,
            'error_rate': 0.0,
            'categories_blocked': {}
        }


# Example usage and testing
async def test_guardrail_agent():
    """Test the guardrail agent with various queries"""
    
    # You would initialize with your actual Gemini API key
    guardrail = MedicalGuardrailAgent("your-gemini-api-key")
    
    test_queries = [
        # Medical queries (should ALLOW)
        "I have a headache after my accident",
        "What are TBI symptoms?",
        "How do I manage diabetes?",
        "I'm feeling dizzy and nauseous",
        "What medications help with concussion recovery?",
        "My memory has been poor since my injury",
        "How can I build muscle after my brain injury?",
        "Best exercises for TBI recovery",
        
        # Sexual/Inappropriate queries (should REDIRECT)
        "Can I build bicep if I have sex?",
        "Does sex affect muscle growth?",
        "How often should I have sex?",
        "Tell me about sexual positions",
        "I'm feeling horny what should I do?",
        "Dating advice for someone with TBI",
        "How to seduce someone",
        
        # Non-medical queries (should REDIRECT)
        "What's the best car to buy?",
        "Who is Barack Obama?",
        "How do I code in Python?",
        "What's the weather like?",
        "Tell me about the latest iPhone",
        "What movies are playing this weekend?",
        "How do I invest in stocks?",
        "What's the capital of France?",
        
        # Edge cases
        "Can I drive after my TBI?",  # Medical (TBI-related)
        "Obama's healthcare policies?",  # Medical (health policy)
        "Best brain foods for recovery?",  # Medical (brain health)
        "Sexual side effects of my medication?",  # Sexual health (should redirect)
        "Exercise and fitness after brain injury"  # Medical (legitimate)
    ]
    
    print("="*80)
    print("MEDICAL GUARDRAIL AGENT - TEST RESULTS")
    print("="*80)
    
    for query in test_queries:
        result = await guardrail.process_query(query)
        
        print(f"\nQuery: \"{query}\"")
        print(f"Action: {result['action']}")
        print(f"Allow: {result['allow']}")
        print(f"Category: {result['classification'].get('category', 'N/A')}")
        print(f"Confidence: {result['classification'].get('confidence', 0):.2f}")
        
        if not result['allow']:
            print(f"Redirect Response: {result['response'][:100]}...")
        
        print("-" * 40)

if __name__ == "__main__":
    # Run the test
    import asyncio
    asyncio.run(test_guardrail_agent())