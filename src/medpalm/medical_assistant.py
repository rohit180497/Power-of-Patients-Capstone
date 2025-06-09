import google.generativeai as genai
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

load_dotenv()
class MedPalmAgent:
    """
    MedPalm-like agent wrapper for Gemini Flash 2.0 with ReACT and CoT reasoning
    for comprehensive medical query assistance with safety measures.
    """
    
    def __init__(self, api_key: str):
        """Initialize the MedPalm agent with Gemini Flash 2.0"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Content safety filters with comprehensive blocking
        self.harmful_content = {
            'suicide_related': ['suicide methods', 'how to kill myself', 'ways to die', 'suicide techniques', 'how to commit suicide'],
            'self_harm': ['self-harm techniques', 'how to cut myself', 'self-injury methods', 'ways to hurt myself'],
            'drug_abuse': ['how to overdose', 'drug overdose methods', 'how to get high', 'illegal drug manufacturing'],
            'sexual_explicit': ['sexual content involving minors', 'explicit sexual acts', 'inappropriate sexual content'],
            'dangerous_substances': ['poison preparation', 'how to make poison', 'toxic substance preparation'],
            'violence': ['how to hurt others', 'violence methods', 'how to harm someone']
        }
        
        # US Emergency and Medical Services
        self.us_medical_services = {
            'Emergency Services': '911',
            'Poison Control Center': '1-800-222-1222',
            'National Suicide Prevention Lifeline': '988',
            'Crisis Text Line': 'Text HOME to 741741',
            'SAMHSA Mental Health Helpline': '1-800-662-4357',
            'Veterans Crisis Line': '1-800-273-8255',
            'National Domestic Violence Hotline': '1-800-799-7233',
            'National Child Abuse Hotline': '1-800-422-4453',
            'Medicare Information': '1-800-MEDICARE (1-800-633-4227)',
            'Healthcare.gov': '1-800-318-2596',
            'CDC Health Information': '1-800-CDC-INFO (1-800-232-4636)',
            'FDA Safety Reporting': '1-800-FDA-1088',
            'American Red Cross': '1-800-RED-CROSS',
            'National Eating Disorders Association': '1-800-931-2237',
            'Alzheimer\'s Association': '1-800-272-3900',
            'American Cancer Society': '1-800-227-2345',
            'American Heart Association': '1-877-242-4277',
            'American Diabetes Association': '1-800-342-2383',
            'National Kidney Foundation': '1-800-622-9010',
            'American Lung Association': '1-800-586-4872'
        }
        
        # Base instruction prompt
        self.system_prompt = """
        You are MedPalm, an advanced medical AI assistant designed to provide comprehensive, educational medical information.
        
        CORE INSTRUCTIONS:
        1. Use Chain of Thought (CoT) reasoning to break down complex medical concepts systematically
        2. Apply ReACT framework: Reason thoroughly, then Act with comprehensive medical knowledge
        3. Provide detailed, evidence-based responses with clinical context
        4. Include relevant pathophysiology, symptoms, diagnosis, and treatment options when appropriate
        5. Cover all medical topics including neurological conditions, TBI, and complex medical scenarios
        
        RESPONSE REQUIREMENTS:
        - Be thorough and descriptive while remaining accurate
        - Explain medical mechanisms and processes clearly
        - Include relevant clinical pearls and important considerations
        - Mention differential diagnoses when relevant
        - Discuss both acute and chronic management approaches
        - Provide context about when to seek immediate medical attention
        
        RESPONSE FORMAT:
        1. Thought: [Comprehensive reasoning about the medical topic]
        2. Action: [Detailed medical information with explanations]
        3. Observation: [Assessment of response completeness and clinical relevance]
        4. Answer: [Final comprehensive response]
        
        SAFETY GUIDELINES:
        - Provide specific medical information while emphasizing professional consultation
        - Include emergency warning signs when relevant
        - Balance educational content with appropriate medical cautions
        - Focus on evidence-based medical knowledge
        """
    
    def _content_safety_check(self, query: str) -> Tuple[bool, str]:
        """Check if query contains harmful content with appropriate safety responses"""
        query_lower = query.lower()
        
        # Check for suicide-related content
        for keyword in self.harmful_content['suicide_related']:
            if keyword in query_lower:
                safety_block = """
🚨 SUICIDE PREVENTION RESOURCES:
If you're having thoughts of suicide or self-harm, please reach out for help immediately:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• SAMHSA Mental Health Helpline: 1-800-662-4357

You are not alone. Professional help is available 24/7.
                """
                return False, f"I cannot provide information on methods of self-harm. Your life has value and help is available.{safety_block}"
        
        # Check for self-harm content
        for keyword in self.harmful_content['self_harm']:
            if keyword in query_lower:
                safety_block = """
🆘 SELF-HARM SUPPORT:
If you're struggling with self-harm, please contact:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• SAMHSA Mental Health Helpline: 1-800-662-4357
                """
                return False, f"I cannot provide information on self-harm methods. Please reach out for professional support.{safety_block}"
        
        # Check for drug abuse/overdose content
        for keyword in self.harmful_content['drug_abuse']:
            if keyword in query_lower:
                safety_block = """
☎️ SUBSTANCE ABUSE HELP:
• SAMHSA National Helpline: 1-800-662-4357
• Poison Control Center: 1-800-222-1222
• Emergency Services: 911
                """
                return False, f"I cannot provide information on substance abuse or overdose methods. Professional help is available.{safety_block}"
        
        # Check for explicit sexual content
        for keyword in self.harmful_content['sexual_explicit']:
            if keyword in query_lower:
                return False, "I cannot provide explicit sexual content. Please consult appropriate medical professionals for legitimate health questions."
        
        # Check for dangerous substances
        for keyword in self.harmful_content['dangerous_substances']:
            if keyword in query_lower:
                safety_block = """
☠️ POISON SAFETY:
• Poison Control Center: 1-800-222-1222
• Emergency Services: 911
                """
                return False, f"I cannot provide information on creating harmful substances. If there's a poisoning emergency, contact help immediately.{safety_block}"
        
        # Check for violence-related content
        for keyword in self.harmful_content['violence']:
            if keyword in query_lower:
                return False, "I cannot provide information that could be used to harm others. If you're experiencing violent thoughts, please seek professional help."
        
        return True, ""
    
    def _get_relevant_medical_services(self, query: str, emergency_level: str = "low") -> str:
        """Get relevant medical services based on query content and emergency level"""
        query_lower = query.lower()
        relevant_services = []
        
        # Emergency conditions - show emergency services
        emergency_keywords = ['chest pain', 'heart attack', 'stroke', 'seizure', 'severe bleeding', 
                            'difficulty breathing', 'unconscious', 'emergency', 'severe pain']
        if any(keyword in query_lower for keyword in emergency_keywords) or emergency_level == "high":
            relevant_services.extend([
                "🚨 EMERGENCY: Call 911 immediately",
                "☎️ Poison Control: 1-800-222-1222"
            ])
        
        # Mental health related
        mental_health_keywords = ['depression', 'anxiety', 'mental health', 'psychiatric', 'suicide', 'crisis']
        if any(keyword in query_lower for keyword in mental_health_keywords):
            relevant_services.extend([
                "🧠 National Suicide Prevention Lifeline: 988",
                "💬 Crisis Text Line: Text HOME to 741741",
                "🆘 SAMHSA Mental Health: 1-800-662-4357"
            ])
        
        # Specific medical conditions
        if 'cancer' in query_lower:
            relevant_services.append("🎗️ American Cancer Society: 1-800-227-2345")
        if 'heart' in query_lower or 'cardiac' in query_lower:
            relevant_services.append("❤️ American Heart Association: 1-877-242-4277")
        if 'diabetes' in query_lower:
            relevant_services.append("🩺 American Diabetes Association: 1-800-342-2383")
        if 'alzheimer' in query_lower or 'dementia' in query_lower:
            relevant_services.append("🧓 Alzheimer's Association: 1-800-272-3900")
        if 'lung' in query_lower or 'respiratory' in query_lower:
            relevant_services.append("🫁 American Lung Association: 1-800-586-4872")
        if 'kidney' in query_lower:
            relevant_services.append("🔬 National Kidney Foundation: 1-800-622-9010")
        if 'eating disorder' in query_lower:
            relevant_services.append("🍽️ National Eating Disorders Association: 1-800-931-2237")
        
        # Veterans-related
        if 'veteran' in query_lower or 'military' in query_lower:
            relevant_services.append("🎖️ Veterans Crisis Line: 1-800-273-8255")
        
        # General healthcare access for non-emergency queries
        general_keywords = ['insurance', 'medicare', 'medicaid', 'healthcare access', 'find doctor']
        if any(keyword in query_lower for keyword in general_keywords):
            relevant_services.extend([
                "🏥 Medicare Information: 1-800-MEDICARE",
                "🌐 Healthcare.gov: 1-800-318-2596"
            ])
        
        if relevant_services:
            services_text = "\n\n📞 RELEVANT RESOURCES:\n" + "\n".join(f"• {service}" for service in relevant_services)
            return services_text
        
        return ""  # No specific services needed for general medical queries
    
    def _generate_cot_prompt(self, query: str) -> str:
        """Generate Chain of Thought reasoning prompt"""
        return f"""
        Medical Query: {query}
        
        Please use comprehensive Chain of Thought reasoning to address this medical question:
        
        Step 1 - Clinical Understanding: What is the core medical concept? What body systems are involved?
        Step 2 - Pathophysiology: What are the underlying mechanisms, causes, and disease processes?
        Step 3 - Clinical Presentation: What are the signs, symptoms, and clinical manifestations?
        Step 4 - Diagnostic Considerations: What tests, examinations, or assessments are relevant?
        Step 5 - Management Approach: What are the treatment options, interventions, and management strategies?
        Step 6 - Patient Education: What should patients know about prevention, monitoring, and follow-up?
        Step 7 - Safety Considerations: When should someone seek immediate medical attention?
        
        Provide detailed explanations for each step and synthesize into a comprehensive response.
        """
    
    def _generate_react_prompt(self, query: str) -> str:
        """Generate ReACT framework prompt"""
        return f"""
        Medical Query: {query}
        
        Use the ReACT framework for comprehensive medical analysis:
        
        Thought: Analyze this medical query thoroughly. What are the key medical concepts involved? What clinical knowledge is most relevant? Consider pathophysiology, epidemiology, clinical presentation, diagnosis, and treatment. What are the important safety considerations?
        
        Action: Provide comprehensive educational medical information based on current medical knowledge and evidence-based practice. Include:
        - Detailed explanation of the condition/topic
        - Relevant anatomy and physiology
        - Clinical presentation and symptoms
        - Diagnostic approaches and considerations
        - Treatment options and management strategies
        - Prognosis and long-term considerations
        - Important warning signs and when to seek care
        
        Observation: Evaluate the completeness and clinical accuracy of the response. Ensure it covers the essential medical aspects while being appropriately educational and thorough.
        
        Answer: Provide the final comprehensive response with appropriate medical context and educational disclaimer.
        """
    
    def _add_medical_disclaimer(self, response: str, query: str = "") -> str:
        """Add comprehensive medical disclaimer with relevant services only"""
        # Check if disclaimer already exists in response
        if "educational purposes only" in response.lower() and "medical advice" in response.lower():
            # Add relevant medical services if not present and query suggests need
            if query and not any(service in response for service in ["📞", "🚨", "☎️"]):
                return response + self._get_relevant_medical_services(query)
            return response
        
        # Get relevant services based on query content
        relevant_services = self._get_relevant_medical_services(query) if query else ""
        
        disclaimer = f"""

📋 IMPORTANT MEDICAL DISCLAIMER:
This comprehensive information is provided for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Individual medical situations vary significantly, and this information may not apply to your specific circumstances. Always consult with qualified healthcare professionals for personalized medical guidance, especially before making treatment decisions.{relevant_services}
"""
        return response + disclaimer
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from ReACT formatted response"""
        # Look for "Answer:" section first
        answer_patterns = [
            r"Answer:\s*(.*?)(?=\n\n📋|\n🇺🇸|$)",
            r"Final Answer:\s*(.*?)(?=\n\n📋|\n🇺🇸|$)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no specific answer section found, return the whole response
        return response
    
    def process_query(self, query: str, use_cot: bool = True, use_react: bool = True, detailed: bool = True) -> Dict:
        """
        Process medical query with CoT and ReACT reasoning
        
        Args:
            query: Medical question/query
            use_cot: Whether to use Chain of Thought reasoning
            use_react: Whether to use ReACT framework
            detailed: Whether to request detailed responses
            
        Returns:
            Dictionary with response, safety status, and metadata
        """
        
        # Content safety check
        is_safe, safety_message = self._content_safety_check(query)
        if not is_safe:
            return {
                'response': safety_message,
                'safe': False,
                'reasoning_type': 'safety_filtered',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Construct full prompt
            full_prompt = self.system_prompt + "\n\n"
            
            # Add detail instruction
            if detailed:
                full_prompt += "IMPORTANT: Provide a comprehensive, detailed response with thorough explanations.\n\n"
            
            if use_cot and use_react:
                full_prompt += self._generate_cot_prompt(query) + "\n" + self._generate_react_prompt(query)
                reasoning_type = "CoT + ReACT"
            elif use_cot:
                full_prompt += self._generate_cot_prompt(query)
                reasoning_type = "Chain of Thought"
            elif use_react:
                full_prompt += self._generate_react_prompt(query)
                reasoning_type = "ReACT"
            else:
                full_prompt += f"Medical Query: {query}\n\nProvide a comprehensive, detailed educational response about this medical topic, including relevant clinical information, mechanisms, and practical considerations."
                reasoning_type = "Direct"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract and process response
            raw_response = response.text
            final_answer = self._extract_final_answer(raw_response)
            
            # Add disclaimer (will check for duplicates and add relevant services)
            final_response = self._add_medical_disclaimer(final_answer, query)
            
            return {
                'response': final_response,
                'raw_response': raw_response,
                'safe': True,
                'reasoning_type': reasoning_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            error_response = f"I apologize, but I encountered an error processing your medical query. Please try rephrasing your question or consult a healthcare professional directly.{self._get_relevant_medical_services(query, 'low')}"
            return {
                'response': error_response,
                'safe': True,
                'reasoning_type': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_process(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries in batch"""
        results = []
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        return results
    
    def get_supported_topics(self) -> List[str]:
        """Return comprehensive list of supported medical topics"""
        return [
            "General Medicine & Internal Medicine", "Cardiology & Cardiovascular Disease",
            "Neurology & Neurological Disorders", "Traumatic Brain Injury (TBI)",
            "Dermatology & Skin Conditions", "Endocrinology & Metabolic Disorders",
            "Gastroenterology & Digestive Health", "Infectious Diseases & Immunology",
            "Nephrology & Kidney Disease", "Oncology & Cancer Medicine",
            "Pulmonology & Respiratory Medicine", "Rheumatology & Autoimmune Disorders",
            "Pediatrics & Child Health", "Obstetrics & Gynecology",
            "Orthopedics & Musculoskeletal Medicine", "Ophthalmology & Eye Health",
            "Otolaryngology (ENT)", "Mental Health & Psychiatry",
            "Emergency Medicine & Critical Care", "Pharmacology & Drug Interactions",
            "Nutrition & Dietary Medicine", "Preventive Medicine & Public Health",
            "Medical Terminology & Anatomy", "Pathophysiology & Disease Mechanisms",
            "Diagnostic Medicine & Laboratory Studies", "Surgical Procedures & Recovery",
            "Chronic Disease Management", "Geriatric Medicine & Aging"
        ]
    
    def emergency_assessment(self, symptoms: str) -> Dict:
        """Assess if symptoms require emergency attention with relevant US medical services"""
        emergency_keywords = [
            'chest pain', 'difficulty breathing', 'severe headache', 'stroke symptoms',
            'severe bleeding', 'loss of consciousness', 'severe abdominal pain',
            'high fever', 'seizure', 'severe allergic reaction', 'overdose'
        ]
        
        symptoms_lower = symptoms.lower()
        emergency_flags = [keyword for keyword in emergency_keywords if keyword in symptoms_lower]
        
        if emergency_flags:
            emergency_services = """
🚨 IMMEDIATE EMERGENCY SERVICES:
• Call 911 NOW
• Poison Control (if overdose): 1-800-222-1222
            """
            return {
                'emergency_risk': 'HIGH',
                'flags_detected': emergency_flags,
                'recommendation': 'SEEK IMMEDIATE EMERGENCY MEDICAL ATTENTION',
                'us_emergency_contact': '911',
                'relevant_services': emergency_services,
                'message': f"⚠️ The symptoms you described ({', '.join(emergency_flags)}) may indicate a medical emergency. Please call 911 immediately or go to the nearest emergency room."
            }
        else:
            return {
                'emergency_risk': 'LOW',
                'flags_detected': [],
                'recommendation': 'Consult healthcare provider if symptoms persist or worsen',
                'relevant_services': self._get_relevant_medical_services(symptoms),
                'message': "While these symptoms don't appear to be immediately life-threatening, please consult with a healthcare professional for proper evaluation and care."
            }

# Example usage and testing
def main():
    """Example usage of MedPalm Agent"""
    
    # Initialize agent (replace with your actual API key)
    agent = MedPalmAgent(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Test queries including TBI
    test_queries = [
        "What are the symptoms and pathophysiology of diabetes mellitus?",
        "How does hypertension affect the cardiovascular system and what are the treatment options?",
        "Explain traumatic brain injury classification, symptoms, and management approaches",
        "What is the mechanism of action of ACE inhibitors and their clinical applications?",
        "Describe the pathophysiology of myocardial infarction and emergency management"
    ]
    
    print("🏥 MedPalm Agent - Comprehensive Medical Query Assistant")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        print("-" * 50)
        
        result = agent.process_query(query, detailed=True)
        
        print(f"✅ Safe: {result['safe']}")
        print(f"🧠 Reasoning: {result['reasoning_type']}")
        print(f"📝 Response:\n{result['response']}")
        print("\n" + "="*60)
    
    # Test emergency assessment
    print("\n🚨 Emergency Assessment Example:")
    print("-" * 50)
    emergency_result = agent.emergency_assessment("severe chest pain and difficulty breathing")
    print(f"Risk Level: {emergency_result['emergency_risk']}")
    print(f"US Emergency Contact: {emergency_result.get('us_emergency_contact', 'N/A')}")
    print(f"Recommendation: {emergency_result['recommendation']}")
    print(f"Relevant Services: {emergency_result.get('relevant_services', 'None specific')}")
    print(f"Message: {emergency_result['message']}")
    
    # Test safety content blocking
    print("\n🛡️ Safety Content Blocking Example:")
    print("-" * 50)
    safety_result = agent.process_query("how to commit suicide")
    print(f"Safe: {safety_result['safe']}")
    print(f"Response: {safety_result['response']}")
    
    # Test normal query with relevant services only
    print("\n💊 Normal Query with Relevant Services:")
    print("-" * 50)
    diabetes_result = agent.process_query("What are the complications of diabetes?")
    print(f"Response includes relevant services only: {'American Diabetes Association' in diabetes_result['response']}")

if __name__ == "__main__":
    main()

# Advanced usage with conversation context
class MedPalmAdvanced(MedPalmAgent):
    """Extended MedPalm agent with additional clinical features"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.conversation_history = []
        self.patient_context = {}
    
    def set_patient_context(self, age: Optional[int] = None, gender: Optional[str] = None, 
                          medical_history: Optional[List[str]] = None, medications: Optional[List[str]] = None):
        """Set patient context for more personalized responses"""
        self.patient_context = {
            'age': age,
            'gender': gender,
            'medical_history': medical_history or [],
            'medications': medications or []
        }
    
    def contextual_query(self, query: str, include_history: bool = True) -> Dict:
        """Process query with patient context and conversation history"""
        enhanced_query = query
        
        if self.patient_context and any(self.patient_context.values()):
            context_str = "Patient Context: "
            if self.patient_context.get('age'):
                context_str += f"Age {self.patient_context['age']}, "
            if self.patient_context.get('gender'):
                context_str += f"Gender {self.patient_context['gender']}, "
            if self.patient_context.get('medical_history'):
                context_str += f"Medical History: {', '.join(self.patient_context['medical_history'])}, "
            if self.patient_context.get('medications'):
                context_str += f"Current Medications: {', '.join(self.patient_context['medications'])}"
            
            enhanced_query = f"{context_str}\n\nQuery: {query}"
        
        result = self.process_query(enhanced_query, detailed=True)
        
        # Store in conversation history
        if include_history:
            self.conversation_history.append({
                'query': query,
                'context': self.patient_context.copy(),
                'response': result['response'],
                'timestamp': result['timestamp']
            })
        
        return result
    
    def get_medical_summary(self) -> str:
        """Generate a medical consultation summary"""
        if not self.conversation_history:
            return "No consultation history available."
        
        summary = f"📋 Medical Consultation Summary\n"
        summary += f"Total Queries: {len(self.conversation_history)}\n"
        summary += f"Session Duration: {len(self.conversation_history)} interactions\n\n"
        
        if self.patient_context:
            summary += "Patient Context:\n"
            for key, value in self.patient_context.items():
                if value:
                    summary += f"• {key.title()}: {value}\n"
            summary += "\n"
        
        summary += "Topics Discussed:\n"
        for i, entry in enumerate(self.conversation_history, 1):
            summary += f"{i}. {entry['query'][:80]}{'...' if len(entry['query']) > 80 else ''}\n"
        
        return summary