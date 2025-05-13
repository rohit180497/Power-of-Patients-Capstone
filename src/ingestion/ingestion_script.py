import json
import os
import argparse
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pinecone
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Import LlamaIndex components
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

# Import TBI-Optimized Semantic Splitter
from semantic_splitter import TBIOptimizedSemanticSplitter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TBI-Ingestion")

# Constants
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fast and high-quality embedding model
PINECONE_DIMENSION = 384  # For bge-small-en-v1.5


def parse_arguments():
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(description="Ingest TBI articles into Pinecone")
    
    # Input/output parameters
    parser.add_argument("--input_file", type=str, 
                        default=os.getenv('INPUT_FILE'),
                        required=not bool(os.getenv('INPUT_FILE')),
                        help="Path to simplified_all_articles.json (can be set via INPUT_FILE env var)")
    parser.add_argument("--output_dir", type=str, 
                        default=os.getenv('OUTPUT_DIR', "./tbi_ingestion_output"),
                        help="Directory to save output files (can be set via OUTPUT_DIR env var)")
    
    # Pinecone parameters
    parser.add_argument("--pinecone_api_key", type=str, 
                        default=os.getenv('PINECONE_API_KEY'),
                        required=not bool(os.getenv('PINECONE_API_KEY')),
                        help="Pinecone API key (can be set via PINECONE_API_KEY env var)")
    parser.add_argument("--pinecone_environment", type=str, 
                        default=os.getenv('PINECONE_ENVIRONMENT'),
                        required=not bool(os.getenv('PINECONE_ENVIRONMENT')),
                        help="Pinecone environment (can be set via PINECONE_ENVIRONMENT env var)")
    parser.add_argument("--index_name", type=str, 
                        default=os.getenv('PINECONE_INDEX_NAME', "power-of-patients-tbi"),
                        help="Pinecone index name (can be set via PINECONE_INDEX_NAME env var)")
    
    # Chunking parameters
    parser.add_argument("--chunk_size", type=int, 
                        default=int(os.getenv('CHUNK_SIZE', DEFAULT_CHUNK_SIZE)),
                        help="Maximum chunk size in characters (can be set via CHUNK_SIZE env var)")
    parser.add_argument("--buffer_size", type=int, 
                        default=int(os.getenv('BUFFER_SIZE', 2)),
                        help="Buffer size for semantic splitter context window (can be set via BUFFER_SIZE env var)")
    parser.add_argument("--breakpoint_threshold", type=int, 
                        default=int(os.getenv('BREAKPOINT_THRESHOLD', 90)),
                        help="Breakpoint percentile threshold (lower=more chunks) (can be set via BREAKPOINT_THRESHOLD env var)")
    
    # Embedding parameters
    embedding_choices = ["openai", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]
    parser.add_argument("--embedding_model", type=str, 
                        default=os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL),
                        choices=embedding_choices,
                        help="Embedding model to use (can be set via EMBEDDING_MODEL env var)")
    
    # Upload parameters
    parser.add_argument("--batch_size", type=int, 
                        default=int(os.getenv('BATCH_SIZE', 100)),
                        help="Batch size for uploading to Pinecone (can be set via BATCH_SIZE env var)")
    
    return parser.parse_args()


def load_articles(filepath: str) -> List[Dict[str, Any]]:
    """Load articles from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    logger.info(f"Loaded {len(articles)} articles from {filepath}")
    return articles

def articles_to_documents(articles: List[Dict[str, Any]]) -> List[Document]:
    """Convert articles to LlamaIndex Document objects."""
    documents = []
    
    for article in articles:
        # Extract relevant fields
        article_id = article.get("id", "")
        url = article.get("url", "")
        title = article.get("title", "")
        date = article.get("date", "")
        author = article.get("author", "")
        read_time = article.get("read_time", "")
        content = article.get("content", "")
        
        # Extract medical categories from title and content
        categories = extract_medical_categories(title, content)
        
        # Create metadata
        metadata = {
            "article_id": article_id,
            "url": url,
            "title": title,
            "date": date,
            "author": author,
            "read_time": read_time,
            "categories": categories
        }
        
        # Create Document object
        doc = Document(
            text=content,
            metadata=metadata,
            doc_id=f"article_{article_id}"
        )
        
        documents.append(doc)
    
    logger.info(f"Converted {len(documents)} articles to Document objects")
    return documents

def extract_medical_categories(title: str, content: str) -> List[str]:
    """Extract medical categories based on content analysis.
    
    This helps with filtering and retrieval by categorizing articles.
    """
    categories = []
    title_and_content = (title + " " + content[:1000]).lower()
    
    # Check for TBI-related content
    if any(term in title_and_content for term in ["traumatic brain injury", "tbi", "brain injury"]):
        categories.append("tbi")
    
    # Check for concussion-specific content
    if "concussion" in title_and_content:
        categories.append("concussion")
    
    # Check for balance-related content
    if any(term in title_and_content for term in ["balance", "vestibular", "dizziness", "equilibrium"]):
        categories.append("balance")
    
    # Check for treatment-related content
    if any(term in title_and_content for term in ["treatment", "therapy", "rehabilitation", "recovery"]):
        categories.append("treatment")
    
    # Check for diagnostic content
    if any(term in title_and_content for term in ["diagnosis", "assessment", "evaluation", "test"]):
        categories.append("diagnosis")
    
    # Check for symptom-related content
    if any(term in title_and_content for term in ["symptom", "headache", "pain", "nausea", "sensitivity"]):
        categories.append("symptoms")
    
    # Default category if none matched
    if not categories:
        categories.append("general")
    
    return categories

def setup_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> BaseEmbedding:
    """Set up and return the embedding model."""
    # OpenAI embedding model - high quality but costs money
    if model_name == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            embed_batch_size=100
        )
        dimension = 1536
    # HuggingFace embedding models - free but may be slower
    elif model_name == "BAAI/bge-small-en-v1.5":
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=32
        )
        dimension = 384
    elif model_name == "BAAI/bge-base-en-v1.5":
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=16
        )
        dimension = 768
    elif model_name == "BAAI/bge-large-en-v1.5":
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=8
        )
        dimension = 1024
    else:
        # Default to bge-small if model not recognized
        logger.warning(f"Model {model_name} not recognized, using BAAI/bge-small-en-v1.5")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=32
        )
        dimension = 384
        
    logger.info(f"Using embedding model: {model_name} with dimension {dimension}")
    return embed_model, dimension


def initialize_pinecone(api_key: str, environment: str, index_name: str, dimension: int) -> Any:
    """
    Initialize Pinecone index using the new Pinecone class
    """
    try:        
        # Create Pinecone client instance
        pc = Pinecone(api_key=api_key)
        
        # Check existing indexes
        existing_indexes = pc.list_indexes()
        
        # Check if index exists
        index_exists = any(idx.name == index_name for idx in existing_indexes)
        
        if not index_exists:
            logger.info(f"Creating new Pinecone index: {index_name} with dimension {dimension}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        # Connect to the index
        index = pc.Index(index_name)
        return index
    
    except Exception as e:
        logger.error(f"Pinecone initialization error: {e}")
        raise


def chunk_documents_with_tbi_splitter(
    documents: List[Document],
    embed_model: BaseEmbedding,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    buffer_size: int = 2,
    breakpoint_threshold: int = 90,
) -> List[Document]:
    """Chunk documents using TBI-optimized semantic splitting."""
    
    logger.info(f"Using TBI-Optimized Semantic Splitter with chunk size {chunk_size}")
    splitter = TBIOptimizedSemanticSplitter(
        embed_model=embed_model,
        buffer_size=buffer_size,  # Window size for context
        breakpoint_percentile_threshold=breakpoint_threshold,  # Adjust for more/fewer chunks
        embedding_length=chunk_size,
        # Increased importance weight for medical terminology
        importance_weight=0.4
    )
    
    all_nodes = []
    for doc in tqdm(documents, desc="Chunking documents"):
        nodes = splitter.get_nodes_from_documents([doc])
        all_nodes.extend(nodes)
    
    logger.info(f"Created {len(all_nodes)} chunks from {len(documents)} documents")
    return all_nodes

def upload_to_pinecone(index, nodes, batch_size=100, save_progress=True, output_dir="."):
    """Upload nodes to Pinecone in batches with progress tracking."""
    total_nodes = len(nodes)
    batches = [nodes[i:i+batch_size] for i in range(0, total_nodes, batch_size)]
    
    logger.info(f"Uploading {total_nodes} nodes to Pinecone in {len(batches)} batches")
    
    start_time = time.time()
    progress_file = os.path.join(output_dir, "upload_progress.json")
    progress_data = {
        "total_nodes": total_nodes,
        "completed_nodes": 0,
        "start_time": start_time,
        "last_update": start_time,
        "batch_details": []
    }
    
    for i, batch in enumerate(tqdm(batches, desc="Uploading to Pinecone")):
        vectors = []
        batch_start = time.time()
        
        for node in batch:
            # Extract embedding
            embedding = node.embedding
            
            # Get metadata from the node
            metadata = node.metadata
            
            # Add text to metadata (truncated)
            metadata["text"] = node.text[:1000] if len(node.text) > 1000 else node.text
            
            # Ensure all metadata values are strings for Pinecone
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ", ".join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            # Create vector record
            vectors.append({
                "id": node.node_id,
                "values": embedding,
                "metadata": metadata
            })
        
        # Upsert batch to Pinecone
        index.upsert(vectors=vectors)
        
        # Update progress
        batch_end = time.time()
        progress_data["completed_nodes"] += len(batch)
        progress_data["last_update"] = batch_end
        progress_data["batch_details"].append({
            "batch": i + 1,
            "size": len(batch),
            "duration": batch_end - batch_start
        })
        
        # Save progress
        if save_progress:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    end_time = time.time()
    logger.info(f"Upload completed in {end_time - start_time:.2f} seconds")
    
    # Save final upload status
    progress_data["end_time"] = end_time
    progress_data["total_duration"] = end_time - start_time
    with open(os.path.join(output_dir, "upload_summary.json"), 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    return progress_data

def save_processed_nodes(nodes, output_dir, filename="processed_nodes.json"):
    """Save processed nodes to a JSON file for reference."""
    simplified_nodes = []
    
    for node in nodes:
        # Create a simplified representation
        simplified_node = {
            "node_id": node.node_id,
            "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            "metadata": node.metadata,
            "embedding_size": len(node.embedding) if node.embedding is not None else 0,
        }
        simplified_nodes.append(simplified_node)
    
    # Save to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_nodes, f, indent=2)
    
    logger.info(f"Saved processed nodes info to {output_path}")
    return output_path

def main():
    # Parse arguments (now with environment variable support)
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure file handler to write to output directory
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "ingestion.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log start of ingestion
    logger.info("Starting Power of Patients TBI article ingestion")
    logger.info(f"Arguments: {args}")
    
    # Load articles
    articles = load_articles(args.input_file)
    
    # Convert to documents
    documents = articles_to_documents(articles)
    
    # Setup embedding model
    embed_model, dimension = setup_embedding_model(args.embedding_model)
    
    # Initialize Pinecone
    index = initialize_pinecone(
        api_key=args.pinecone_api_key,
        environment=args.pinecone_environment,
        index_name=args.index_name,
        dimension=dimension
    )
    
    # Chunk documents with TBI-optimized splitter
    nodes = chunk_documents_with_tbi_splitter(
        documents,
        embed_model=embed_model,
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
        breakpoint_threshold=args.breakpoint_threshold
    )
    
    # Save node information
    save_processed_nodes(nodes, args.output_dir)
    
    # Generate embeddings (if not already done)
    if any(node.embedding is None for node in nodes):
        logger.info("Generating embeddings for chunks...")
        for node in tqdm(nodes, desc="Embedding chunks"):
            if node.embedding is None:
                embedding = embed_model.get_text_embedding(node.text)
                node.embedding = embedding
    
    # Upload to Pinecone
    upload_to_pinecone(
        index, 
        nodes, 
        batch_size=args.batch_size, 
        save_progress=True,
        output_dir=args.output_dir
    )
    
    logger.info(f"Successfully ingested {len(documents)} articles as {len(nodes)} chunks to Pinecone index {args.index_name}")

if __name__ == "__main__":
    main()