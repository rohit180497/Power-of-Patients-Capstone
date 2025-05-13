import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieval_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TBI-Retrieval-Agent")

# Constants
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 5
QUERY_LOG_FILE = os.getenv('QUERY_LOG_FILE', 'query_history.jsonl')

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east1-aws')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'power-of-patients-tbi')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL)

# Global variables to cache connections
_embedding_model = None
_pinecone_index = None

def setup_embedding_model():
    """Set up and cache the embedding model."""
    global _embedding_model
    
    # Return cached model if available
    if _embedding_model is not None:
        return _embedding_model
    
    model_name = EMBEDDING_MODEL
    logger.info(f"Setting up embedding model: {model_name}")
    
    # HuggingFace embedding models
    if model_name == "BAAI/bge-small-en-v1.5":
        _embedding_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=32
        )
    elif model_name == "BAAI/bge-base-en-v1.5":
        _embedding_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=16
        )
    elif model_name == "BAAI/bge-large-en-v1.5":
        _embedding_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=8
        )
    else:
        # Default to bge-small if model not recognized
        logger.warning(f"Model {model_name} not recognized, using BAAI/bge-small-en-v1.5")
        _embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=32
        )
        
    logger.info(f"Embedding model setup complete: {model_name}")
    return _embedding_model

def connect_to_pinecone():
    """Connect to Pinecone index and cache the connection."""
    global _pinecone_index
    
    # Return cached connection if available
    if _pinecone_index is not None:
        return _pinecone_index
    
    # Check for API key
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
    
    logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME} in environment: {PINECONE_ENVIRONMENT}")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        indexes = pc.list_indexes()
        
        if not any(idx.name == PINECONE_INDEX_NAME for idx in indexes):
            available_indexes = [idx.name for idx in indexes]
            raise ValueError(f"Index {PINECONE_INDEX_NAME} not found. Available indexes: {available_indexes}")
        
        # Connect to the index
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        return _pinecone_index
    
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        raise

def generate_article_summary(text: str, max_length: int = 250) -> str:
    """Generate a brief 2-3 line summary of an article text."""
    if not text:
        return ""
        
    # Split into sentences and build summary
    sentences = text.split('.')
    summary_sentences = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Add period back to sentence
        sentence_clean = sentence.strip() + '.'
        
        # Add sentence if it won't exceed max length
        if current_length + len(sentence_clean) <= max_length:
            summary_sentences.append(sentence_clean)
            current_length += len(sentence_clean)
        else:
            # If we already have at least 2 sentences, stop
            if len(summary_sentences) >= 2:
                break
            # Otherwise, truncate this sentence to fit
            remaining_space = max_length - current_length - 3  # space for ellipsis
            if remaining_space > 30:  # Only add if we can fit a meaningful portion
                truncated = sentence_clean[:remaining_space] + '...'
                summary_sentences.append(truncated)
            break
    
    # If we still have no sentences, just truncate the text
    if not summary_sentences and text:
        return text[:max_length-3] + '...'
        
    summary = ' '.join(summary_sentences)
    
    # Ensure summary doesn't exceed max length
    if len(summary) > max_length:
        summary = summary[:max_length-3] + '...'
    
    return summary

def log_query_and_results(query: str, results: List[Dict[str, Any]], user_id: Optional[str] = None):
    """
    Log the query and results with timestamp.
    This can be stored in a database with patient ID or doctor ID later.
    
    Args:
        query: The search query
        results: The search results
        user_id: Optional ID of the user (patient or doctor)
    """
    timestamp = datetime.now().isoformat()
    
    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "user_id": user_id,  # This can be None now and filled in later
        "results": results
    }
    
    # Append to log file (JSONL format for easy database import)
    try:
        with open(QUERY_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        logger.info(f"Query and results logged at {timestamp}")
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")

def retrieve_articles(
    query: str, 
    top_k: int = DEFAULT_TOP_K, 
    filter_params: Dict[str, Any] = None,
    user_id: Optional[str] = None  # Can be patient_id or doctor_id
) -> List[Dict[str, Any]]:
    """
    Main retrieval function to find articles based on a query.
    
    Args:
        query (str): The search query
        top_k (int): Number of results to return
        filter_params (Dict): Optional filters to apply
        user_id (str): Optional ID of the user (patient or doctor)
        
    Returns:
        List of article dictionaries with summaries, URLs, and read times
    """
    logger.info(f"Retrieving articles for query: '{query}' with top_k={top_k}")
    
    try:
        # Setup embedding model
        embed_model = setup_embedding_model()
        
        # Connect to Pinecone
        index = connect_to_pinecone()
        
        # Generate embedding for query
        start_time = time.time()
        query_embedding = embed_model.get_text_embedding(query)
        logger.info(f"Generated query embedding in {time.time() - start_time:.2f}s")
        
        # Query Pinecone
        query_start_time = time.time()
        results = index.query(
            vector=query_embedding,
            top_k=top_k * 3,  # Request more results to ensure we get enough unique articles
            include_metadata=True,
            filter=filter_params
        )
        logger.info(f"Pinecone query completed in {time.time() - query_start_time:.2f}s")
        
        # Group results by article
        article_matches = {}
        
        for match in results["matches"]:
            article_id = match["metadata"].get("article_id", "")
            
            # Skip if no article ID (shouldn't happen)
            if not article_id:
                continue
                
            # If we haven't seen this article yet, add it
            if article_id not in article_matches:
                article_matches[article_id] = {
                    "id": article_id,
                    "score": match["score"],
                    "title": match["metadata"].get("title", ""),
                    "url": match["metadata"].get("url", ""),
                    "date": match["metadata"].get("date", ""),
                    "author": match["metadata"].get("author", ""),
                    "read_time": match["metadata"].get("read_time", ""),
                    "categories": match["metadata"].get("categories", ""),
                    "best_chunk": match["metadata"].get("text", ""),
                    "chunks": [match["metadata"].get("text", "")]
                }
            else:
                # Update score if this chunk has higher relevance
                if match["score"] > article_matches[article_id]["score"]:
                    article_matches[article_id]["score"] = match["score"]
                    article_matches[article_id]["best_chunk"] = match["metadata"].get("text", "")
                
                # Add this chunk to the list of relevant chunks
                article_matches[article_id]["chunks"].append(match["metadata"].get("text", ""))
        
        # Convert dictionary to list and generate summaries
        matches = []
        for article_id, article in article_matches.items():
            # Generate a summary from the best chunk
            summary = generate_article_summary(article["best_chunk"])
            
            matches.append({
                "id": article["id"],
                "score": article["score"],
                "title": article["title"],
                "summary": summary,
                "url": article["url"],
                "date": article["date"],
                "author": article["author"],
                "read_time": article["read_time"],
                "categories": article["categories"]
            })
        
        # Sort by score and limit to requested top_k
        matches.sort(key=lambda x: x["score"], reverse=True)
        matches = matches[:top_k]
        
        logger.info(f"Retrieved {len(matches)} articles for query")
        
        # Log the query and results
        log_query_and_results(query, matches, user_id)
        
        return matches
    
    except Exception as e:
        logger.error(f"Error in retrieve_articles: {str(e)}")
        # Log the failed query
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user_id": user_id,
            "error": str(e),
            "results": []
        }
        try:
            with open(QUERY_LOG_FILE, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as log_err:
            logger.error(f"Error logging failed query: {str(log_err)}")
            
        return []

def format_results_text(matches: List[Dict[str, Any]]) -> str:
    """Format results as a readable text string."""
    if not matches:
        return "No results found."
        
    output = "\nSearch Results:\n"
    
    for i, match in enumerate(matches, 1):
        output += f"\n[{i}] {match['title']} (Score: {match['score']:.2f})\n"
        output += f"Read time: {match['read_time']}\n"
        output += f"Summary: {match['summary']}\n"
        output += f"URL: {match['url']}\n"
        
        if 'categories' in match and match['categories']:
            output += f"Categories: {match['categories']}\n"
    
    return output

def format_results_json(matches: List[Dict[str, Any]]) -> str:
    """Format results as a JSON string."""
    return json.dumps(matches, indent=2)

# For command line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TBI Knowledge Base Retrieval Agent")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument("--user_id", type=str, help="User ID (patient or doctor)")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Retrieve articles
    results = retrieve_articles(args.query, args.top_k, user_id=args.user_id)
    
    # Output results
    if args.format == "json":
        print(format_results_json(results))
    else:
        print(format_results_text(results))
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                if args.format == "json":
                    f.write(format_results_json(results))
                else:
                    f.write(format_results_text(results))
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving to file: {str(e)}")