#!/usr/bin/env python3
"""
CDC TBI Retrieval & Question Answering System - FINAL VERSION
=============================================================
Retrieve relevant information from CDC TBI knowledge base and generate 
comprehensive, well-sourced answers to TBI-related questions.

Configured for 768-dimensional vectors with BAAI/bge-base-en-v1.5
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dotenv import load_dotenv

# Core dependencies
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
import openai
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CDC-TBI-Retrieval")

class CDCTBIRetriever:
    """CDC TBI Knowledge Base Retrieval System - 768 Dimensions."""
    
    def __init__(self, 
                 pinecone_api_key: str,
                 index_name: str = "us-cdc-tbi",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",  # 768 dimensions
                 llm_provider: str = "gemini"):
        """Initialize the retrieval system."""
        
        self.index_name = index_name
        self.llm_provider = llm_provider
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize embedding model (768 dimensions to match your index)
        self.embed_model = self._setup_embedding_model(embedding_model)
        
        # Initialize LLM
        self._setup_llm()
        
        logger.info(f"Initialized CDC TBI Retriever with index: {index_name}")
        logger.info(f"Using 768-dimensional embeddings with {embedding_model}")
    
    def _setup_embedding_model(self, model_name: str) -> BaseEmbedding:
        """Setup the embedding model - using 768-dimensional model."""
        if model_name == "BAAI/bge-base-en-v1.5":
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=16  # Smaller batch for larger model
            )
            logger.info("✅ Using BAAI/bge-base-en-v1.5 (768 dimensions)")
        elif model_name == "BAAI/bge-small-en-v1.5":
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=32
            )
            logger.warning("⚠️  Using BAAI/bge-small-en-v1.5 (384 dimensions) - may cause dimension mismatch!")
        elif model_name == "BAAI/bge-large-en-v1.5":
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=8
            )
            logger.warning("⚠️  Using BAAI/bge-large-en-v1.5 (1024 dimensions) - may cause dimension mismatch!")
        else:
            # Default to the correct 768-dimensional model
            logger.warning(f"Unknown model {model_name}, defaulting to BAAI/bge-base-en-v1.5")
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-base-en-v1.5",
                embed_batch_size=16
            )
        
        logger.info(f"Loaded embedding model: {model_name}")
        return embed_model
    
    def _setup_llm(self):
        """Setup the language model for answer generation."""
        if self.llm_provider == "gemini":
            # Check for both possible API key names
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.llm_model = "gemini-2.0-flash"
            
        elif self.llm_provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            openai.api_key = api_key
            self.llm_model = "gpt-4-turbo-preview"
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        logger.info(f"Initialized {self.llm_provider} LLM: {self.llm_model}")
    
    def retrieve_context(self, 
                        query: str, 
                        top_k: int = 10,
                        include_metadata: bool = True,
                        filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant context from the CDC TBI knowledge base."""
        
        # Generate query embedding (768 dimensions)
        query_embedding = self.embed_model.get_text_embedding(query)
        
        # Verify embedding dimension
        if len(query_embedding) != 768:
            logger.error(f"Embedding dimension mismatch: got {len(query_embedding)}, expected 768")
            raise ValueError(f"Embedding dimension {len(query_embedding)} does not match index dimension 768")
        
        # Search Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=filter_dict
        )
        
        # Extract and format results
        contexts = []
        for match in search_results['matches']:
            context = {
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'source': {
                    'file_name': match['metadata'].get('file_name', 'Unknown'),
                    'document_category': match['metadata'].get('document_category', 'Unknown'),
                    'publication_year': match['metadata'].get('publication_year', 'Unknown'),
                    'scraped_url': match['metadata'].get('scraped_url', ''),
                    'content_categories': match['metadata'].get('content_categories', ''),
                    'extracted_title': match['metadata'].get('extracted_title', ''),
                }
            }
            contexts.append(context)
        
        logger.info(f"Retrieved {len(contexts)} relevant contexts for query: {query[:50]}...")
        return contexts
    
    def generate_answer(self, 
                       query: str, 
                       contexts: List[Dict],
                       include_sources: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive answer using retrieved contexts."""
        
        # Format contexts for the prompt
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            source_info = f"Source {i}: {ctx['source']['file_name']}"
            if ctx['source']['document_category'] != 'Unknown':
                source_info += f" ({ctx['source']['document_category']})"
            if ctx['source']['publication_year'] != 'Unknown':
                source_info += f" - {ctx['source']['publication_year']}"
            
            formatted_context = f"{source_info}\n{ctx['text']}\n"
            formatted_contexts.append(formatted_context)
        
        context_text = "\n".join(formatted_contexts)
        
        # Create TBI-specific prompt
        prompt = f"""You are a medical AI assistant specializing in traumatic brain injury (TBI) and concussion information. You have access to comprehensive CDC resources on TBI including research publications, clinical guidelines, educational materials, and surveillance data.

USER QUESTION: {query}

RELEVANT CDC INFORMATION:
{context_text}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer based ONLY on the CDC information provided above
2. Structure your response clearly with main points and supporting details
3. Use medical terminology appropriately while remaining accessible
4. Highlight key recommendations, statistics, or clinical guidance when relevant
5. If the question involves treatment or medical advice, emphasize consulting healthcare professionals
6. Include relevant context about TBI severity, populations affected, or prevention when applicable
7. If information is limited or conflicting, acknowledge this appropriately

IMPORTANT GUIDELINES:
- Base your answer strictly on the provided CDC sources
- Do not add information from outside sources
- If the provided information doesn't fully answer the question, state this clearly
- For medical questions, always recommend consulting healthcare professionals
- Maintain a professional, informative tone suitable for both healthcare providers and patients/families

ANSWER:"""

        # Generate response using LLM
        try:
            if self.llm_provider == "gemini":
                model = genai.GenerativeModel(self.llm_model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1500,
                        temperature=0.1,
                    )
                )
                answer_text = response.text
                
            elif self.llm_provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a medical AI assistant specializing in traumatic brain injury information from CDC resources."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1
                )
                answer_text = response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer_text = f"I apologize, but I encountered an error generating the response. Please try again. Error: {str(e)}"
        
        # Compile source information
        sources = []
        if include_sources:
            seen_files = set()
            for ctx in contexts:
                file_name = ctx['source']['file_name']
                if file_name not in seen_files:
                    source = {
                        'file_name': file_name,
                        'document_type': ctx['source']['document_category'],
                        'publication_year': ctx['source']['publication_year'],
                        'relevance_score': round(ctx['score'], 3),
                        'url': ctx['source']['scraped_url'] if ctx['source']['scraped_url'] else None,
                        'title': ctx['source']['extracted_title'] if ctx['source']['extracted_title'] else None
                    }
                    sources.append(source)
                    seen_files.add(file_name)
        
        result = {
            'question': query,
            'answer': answer_text,
            'sources': sources,
            'retrieval_stats': {
                'contexts_retrieved': len(contexts),
                'top_relevance_score': round(contexts[0]['score'], 3) if contexts else 0,
                'generated_at': datetime.now().isoformat(),
                'embedding_dimension': 768
            }
        }
        
        return result
    
    def ask_question(self, 
                    query: str, 
                    top_k: int = 10,
                    include_sources: bool = True,
                    filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Complete question-answering pipeline."""
        
        try:
            # Retrieve relevant contexts
            contexts = self.retrieve_context(query, top_k=top_k, filter_dict=filter_dict)
            
            if not contexts:
                return {
                    'question': query,
                    'answer': "I couldn't find relevant information in the CDC TBI knowledge base to answer your question. Please try rephrasing your question or contact a healthcare professional for medical advice.",
                    'sources': [],
                    'retrieval_stats': {
                        'contexts_retrieved': 0,
                        'top_relevance_score': 0,
                        'generated_at': datetime.now().isoformat(),
                        'embedding_dimension': 768
                    }
                }
            
            # Generate answer
            result = self.generate_answer(query, contexts, include_sources)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ask_question: {e}")
            return {
                'question': query,
                'answer': f"I encountered an error processing your question: {str(e)}. Please try again or contact support.",
                'sources': [],
                'retrieval_stats': {
                    'contexts_retrieved': 0,
                    'top_relevance_score': 0,
                    'generated_at': datetime.now().isoformat(),
                    'embedding_dimension': 768,
                    'error': str(e)
                }
            }
    
    def batch_questions(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple questions in batch."""
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            
            result = self.ask_question(question, **kwargs)
            results.append(result)
            
            # Small delay to be respectful to APIs
            time.sleep(0.5)
        
        return results
    
    def search_by_category(self, 
                          query: str, 
                          category: str, 
                          top_k: int = 5) -> Dict[str, Any]:
        """Search within a specific document category."""
        
        category_filters = {
            'research': {'document_category': {'$in': ['MMWR Article', 'Research Publication', 'Surveillance Report']}},
            'clinical': {'document_category': {'$in': ['Clinical Guideline']}},
            'educational': {'document_category': {'$in': ['Educational Material', 'HEADS UP Resource', 'Fact Sheet']}},
            'policy': {'document_category': {'$in': ['Policy Report']}},
            'prevention': {'document_category': {'$in': ['Prevention Resource']}},
            'pediatric': {'content_categories': {'$in': ['Pediatric']}},
            'sports': {'content_categories': {'$in': ['Sports']}},
            'concussion': {'content_categories': {'$in': ['Concussion']}}
        }
        
        filter_dict = category_filters.get(category.lower())
        if not filter_dict:
            logger.warning(f"Unknown category: {category}")
            filter_dict = None
        
        return self.ask_question(query, top_k=top_k, filter_dict=filter_dict)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats['total_vector_count'],
                'index_fullness': stats.get('index_fullness', 0),
                'dimension': stats.get('dimension', 'Unknown'),
                'namespace_stats': stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CDC TBI Retrieval & QA System (768 Dimensions)")
    
    parser.add_argument("--pinecone_api_key", type=str,
                        default=os.getenv('PINECONE_API_KEY'),
                        help="Pinecone API key")
    parser.add_argument("--index_name", type=str,
                        default=os.getenv('PINECONE_INDEX2_NAME', 'us-cdc-tbi'),
                        help="Pinecone index name")
    parser.add_argument("--embedding_model", type=str,
                        default="BAAI/bge-base-en-v1.5",  # 768 dimensions
                        help="Embedding model (must be 768 dimensions)")
    parser.add_argument("--llm_provider", type=str,
                        default="gemini",
                        choices=["gemini", "openai"],
                        help="LLM provider for answer generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--output_file", type=str,
                        default="tbi_qa_results.json",
                        help="File to save results")
    
    return parser.parse_args()

def run_test_queries(retriever: CDCTBIRetriever) -> List[Dict[str, Any]]:
    """Run comprehensive TBI test queries."""
    
    test_queries = [
        # General TBI Information
        "What is a traumatic brain injury and what causes it?",
        "How common are traumatic brain injuries in the United States?",
        "What are the main types and severity levels of TBI?",
        
        # Symptoms and Diagnosis
        "What are the signs and symptoms of a concussion?",
        "How is traumatic brain injury diagnosed in emergency departments?",
        "What are the danger signs after a head injury that require immediate medical attention?",
        
        # Treatment and Recovery
        "What are the recommended treatments for mild traumatic brain injury?",
        "How long does recovery from a concussion typically take?",
        "What rehabilitation services are available for TBI patients?",
        
        # Prevention
        "How can traumatic brain injuries be prevented?",
        "What safety measures are recommended for sports-related concussion prevention?",
        "What helmet safety guidelines exist for preventing TBI?",
        
        # Specific Populations
        "How does TBI affect children differently than adults?",
        "What are the risks of repeated concussions in sports?",
        "How does TBI impact older adults and what are the unique considerations?",
        
        # Epidemiology and Statistics
        "What are the leading causes of TBI-related deaths in the United States?",
        "How has the incidence of sports-related TBI changed over time?",
        "What demographic groups are at highest risk for traumatic brain injury?",
        
        # Return to Activity
        "When is it safe to return to sports after a concussion?",
        "What are the CDC recommendations for return to school after TBI?",
        "What workplace accommodations might be needed after a brain injury?",
        
        # Long-term Effects
        "What are the potential long-term effects of traumatic brain injury?",
        "How does TBI increase the risk of other health conditions?",
        "What is chronic traumatic encephalopathy and how is it related to repeated head injuries?"
    ]
    
    logger.info(f"Running {len(test_queries)} comprehensive TBI test queries...")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(test_queries)}: {query}")
        print('='*80)
        
        start_time = time.time()
        result = retriever.ask_question(query, top_k=8)
        end_time = time.time()
        
        print(f"\nANSWER:")
        print(result['answer'])
        
        print(f"\nSOURCES ({len(result['sources'])} documents):")
        for j, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
            print(f"{j}. {source['file_name']} ({source['document_type']}) - Relevance: {source['relevance_score']}")
            if source.get('title'):
                print(f"   Title: {source['title'][:100]}...")
        
        print(f"\nQuery Time: {end_time - start_time:.2f} seconds")
        print(f"Retrieved: {result['retrieval_stats']['contexts_retrieved']} contexts")
        print(f"Top Relevance: {result['retrieval_stats']['top_relevance_score']}")
        print(f"Embedding Dimension: {result['retrieval_stats']['embedding_dimension']}")
        
        results.append(result)
        
        # Small delay between questions
        time.sleep(1)
    
    return results

def run_interactive_mode(retriever: CDCTBIRetriever):
    """Run interactive question-answering session."""
    print("\n🧠 CDC TBI Interactive Q&A System (768 Dimensions)")
    print("="*55)
    print("Ask questions about traumatic brain injury based on CDC resources.")
    print("Type 'quit' to exit, 'stats' for knowledge base statistics.")
    print("Prefix with 'category:' to search specific categories (e.g., 'clinical: treatment options')")
    
    while True:
        try:
            query = input("\n💭 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = retriever.get_index_stats()
                print(f"\n📊 Knowledge Base Statistics:")
                print(f"Total documents: {stats.get('total_vectors', 'Unknown')}")
                print(f"Index dimension: {stats.get('dimension', 'Unknown')}")
                continue
            
            if not query:
                continue
            
            # Check for category-specific search
            if ':' in query and query.split(':')[0].lower() in ['research', 'clinical', 'educational', 'policy', 'prevention', 'pediatric', 'sports', 'concussion']:
                category, actual_query = query.split(':', 1)
                result = retriever.search_by_category(actual_query.strip(), category.strip())
                print(f"\n🔍 Searching in category: {category.upper()}")
            else:
                result = retriever.ask_question(query)
            
            print(f"\n💡 Answer:")
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"{i}. {source['file_name']} ({source['document_type']})")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate required API keys
    if args.llm_provider == "gemini":
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini LLM")
            return
    elif args.llm_provider == "openai" and not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY required for OpenAI LLM")
        return
    
    if not args.pinecone_api_key:
        print("❌ PINECONE_API_KEY required")
        return
    
    # Initialize retriever
    print("🔧 Initializing CDC TBI Retrieval System (768 Dimensions)...")
    try:
        retriever = CDCTBIRetriever(
            pinecone_api_key=args.pinecone_api_key,
            index_name=args.index_name,
            embedding_model=args.embedding_model,
            llm_provider=args.llm_provider
        )
        
        # Show knowledge base stats
        stats = retriever.get_index_stats()
        print(f"📊 Knowledge Base: {stats.get('total_vectors', 'Unknown')} documents indexed")
        print(f"📐 Index Dimension: {stats.get('dimension', 'Unknown')}")
        
        if args.interactive:
            run_interactive_mode(retriever)
        else:
            # Run comprehensive test queries
            print("🧪 Running comprehensive TBI test queries...")
            results = run_test_queries(retriever)
            
            # Save results
            output_data = {
                'test_info': {
                    'total_questions': len(results),
                    'index_name': args.index_name,
                    'embedding_model': args.embedding_model,
                    'embedding_dimension': 768,
                    'llm_provider': args.llm_provider,
                    'run_date': datetime.now().isoformat()
                },
                'results': results
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved to {args.output_file}")
            
            # Summary statistics
            total_contexts = sum(r['retrieval_stats']['contexts_retrieved'] for r in results)
            avg_relevance = sum(r['retrieval_stats']['top_relevance_score'] for r in results) / len(results)
            
            print(f"\n📈 Summary Statistics:")
            print(f"Questions processed: {len(results)}")
            print(f"Total contexts retrieved: {total_contexts}")
            print(f"Average relevance score: {avg_relevance:.3f}")
            print(f"Embedding dimension: 768")
            
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()