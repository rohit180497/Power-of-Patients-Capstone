#!/usr/bin/env python3
"""
CDC TBI PDF Ingestion Pipeline
==============================
Ingest CDC TBI PDF collection into Pinecone with rich metadata extraction.
Processes organized PDF collections and maintains full document metadata.
"""

import json
import os
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime
from tqdm import tqdm
import pinecone
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# PDF Processing
import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF

# Import LlamaIndex components
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

# Import TBI-Optimized Semantic Splitter 
try:
    from semantic_splitter import TBIOptimizedSemanticSplitter
except ImportError:
    print("Warning: TBIOptimizedSemanticSplitter not found. Using basic chunking.")
    TBIOptimizedSemanticSplitter = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CDC-TBI-PDF-Ingestion")

# Constants
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

class PDFProcessor:
    """Handle PDF text extraction and metadata parsing."""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pymupdf,
            self._extract_with_pypdf2,
            self._extract_with_pdfminer
        ]
    
    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF using multiple methods."""
        pdf_path = Path(pdf_path)
        
        # Try different extraction methods
        text = ""
        pdf_metadata = {}
        
        for method in self.extraction_methods:
            try:
                text, pdf_metadata = method(pdf_path)
                if text.strip():  # If we got meaningful text, use it
                    logger.debug(f"Successfully extracted text using {method.__name__}")
                    break
            except Exception as e:
                logger.debug(f"Method {method.__name__} failed: {e}")
                continue
        
        if not text.strip():
            logger.warning(f"Could not extract text from {pdf_path}")
            text = f"[PDF content could not be extracted from {pdf_path.name}]"
        
        # Enhance metadata with file information
        enhanced_metadata = self._enhance_metadata(pdf_path, pdf_metadata, text)
        
        return text, enhanced_metadata
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract using pdfplumber (best for tables and complex layouts)."""
        text_parts = []
        metadata = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            # Get PDF metadata
            if pdf.metadata:
                metadata.update({
                    'pdf_title': pdf.metadata.get('Title', ''),
                    'pdf_author': pdf.metadata.get('Author', ''),
                    'pdf_creator': pdf.metadata.get('Creator', ''),
                    'pdf_producer': pdf.metadata.get('Producer', ''),
                    'pdf_creation_date': pdf.metadata.get('CreationDate', ''),
                    'pdf_mod_date': pdf.metadata.get('ModDate', ''),
                })
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}\n")
        
        return "\n".join(text_parts), metadata
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract using PyMuPDF (fast and good for most PDFs)."""
        doc = fitz.open(str(pdf_path))
        text_parts = []
        metadata = doc.metadata
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}\n")
        
        doc.close()
        return "\n".join(text_parts), metadata
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract using PyPDF2 (reliable but basic)."""
        text_parts = []
        metadata = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get metadata
            if pdf_reader.metadata:
                metadata = {
                    'pdf_title': pdf_reader.metadata.get('/Title', ''),
                    'pdf_author': pdf_reader.metadata.get('/Author', ''),
                    'pdf_creator': pdf_reader.metadata.get('/Creator', ''),
                    'pdf_producer': pdf_reader.metadata.get('/Producer', ''),
                    'pdf_creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                    'pdf_mod_date': str(pdf_reader.metadata.get('/ModDate', '')),
                }
            
            # Extract text
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}\n")
        
        return "\n".join(text_parts), metadata
    
    def _extract_with_pdfminer(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Extract using pdfminer (good for complex layouts)."""
        text = extract_text(str(pdf_path))
        return text, {}
    
    def _enhance_metadata(self, pdf_path: Path, pdf_metadata: Dict, text: str) -> Dict:
        """Enhance metadata with file analysis and content extraction."""
        enhanced = {
            'file_name': pdf_path.name,
            'file_path': str(pdf_path),
            'file_size': pdf_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),
            'extraction_date': datetime.now().isoformat()
        }
        
        # Add PDF metadata
        enhanced.update(pdf_metadata)
        
        # Determine document category from file path
        enhanced['document_category'] = self._categorize_from_path(pdf_path)
        
        # Extract information from content
        content_metadata = self._extract_from_content(text)
        enhanced.update(content_metadata)
        
        return enhanced
    
    def _categorize_from_path(self, pdf_path: Path) -> str:
        """Categorize document based on file structure."""
        path_str = str(pdf_path).lower()
        
        if 'research_publications' in path_str:
            if 'mmwr' in path_str:
                return 'MMWR Article'
            elif 'surveillance' in path_str:
                return 'Surveillance Report'
            else:
                return 'Research Publication'
        elif 'clinical_guidelines' in path_str:
            return 'Clinical Guideline'
        elif 'educational_materials' in path_str:
            if 'heads_up' in path_str:
                return 'HEADS UP Resource'
            elif 'fact_sheet' in path_str:
                return 'Fact Sheet'
            else:
                return 'Educational Material'
        elif 'policy_reports' in path_str:
            return 'Policy Report'
        elif 'prevention_resources' in path_str:
            return 'Prevention Resource'
        else:
            return 'CDC TBI Document'
    
    def _extract_from_content(self, text: str) -> Dict:
        """Extract metadata from document content."""
        metadata = {}
        
        # Extract title (usually in the first few lines)
        lines = text.split('\n')[:20]  # Check first 20 lines
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                # Skip lines that look like headers/footers
                if not any(skip in line.lower() for skip in ['page', 'cdc', 'www.', 'http']):
                    metadata['extracted_title'] = line
                    break
        
        # Extract publication year
        year_matches = re.findall(r'\b(19|20)\d{2}\b', text[:2000])
        if year_matches:
            metadata['publication_year'] = year_matches[0]
        
        # Extract DOI
        doi_pattern = r'10\.\d{4,}/[^\s<>\"\']*'
        doi_matches = re.findall(doi_pattern, text)
        if doi_matches:
            metadata['doi'] = doi_matches[0]
        
        # Extract author information (basic patterns)
        author_patterns = [
            r'Authors?:?\s*([A-Z][a-zA-Z\s,.-]+)',
            r'By:?\s*([A-Z][a-zA-Z\s,.-]+)',
            r'([A-Z][a-zA-Z]+,\s*[A-Z][a-zA-Z\s]*\s*(?:MD|PhD|MPH|MS|RN))'
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:1000])
            if matches:
                metadata['extracted_authors'] = matches[0]
                break
        
        # Identify TBI-specific categories
        categories = []
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['traumatic brain injury', 'tbi']):
            categories.append('TBI')
        if 'concussion' in text_lower:
            categories.append('Concussion')
        if any(term in text_lower for term in ['surveillance', 'epidemiology', 'statistics']):
            categories.append('Surveillance')
        if any(term in text_lower for term in ['treatment', 'therapy', 'rehabilitation']):
            categories.append('Treatment')
        if any(term in text_lower for term in ['prevention', 'safety', 'helmet']):
            categories.append('Prevention')
        if any(term in text_lower for term in ['pediatric', 'children', 'youth']):
            categories.append('Pediatric')
        if any(term in text_lower for term in ['sports', 'athletic', 'football']):
            categories.append('Sports')
        
        metadata['content_categories'] = categories
        metadata['content_length'] = len(text)
        metadata['word_count'] = len(text.split())
        
        return metadata

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest CDC TBI PDFs into Pinecone")
    
    # Input parameters
    parser.add_argument("--pdf_directory", type=str, 
                        default=os.getenv('PDF_DIRECTORY', './CDC_TBI_Resources'),
                        help="Root directory containing organized PDF files")
    parser.add_argument("--metadata_file", type=str, 
                        default=os.getenv('METADATA_FILE'),
                        help="Optional: CSV metadata file from scraping")
    parser.add_argument("--output_dir", type=str, 
                        default=os.getenv('OUTPUT_DIR', "./cdc_tbi_ingestion_output"),
                        help="Directory to save output files")
    
    # Pinecone parameters  
    parser.add_argument("--pinecone_api_key", type=str, 
                        default=os.getenv('PINECONE_API_KEY'),
                        required=not bool(os.getenv('PINECONE_API_KEY')),
                        help="Pinecone API key")
    parser.add_argument("--index_name", type=str, 
                        default=os.getenv('PINECONE_INDEX2_NAME', "us-cdc-tbi"),
                        help="Pinecone index name")
    
    # Processing parameters
    parser.add_argument("--chunk_size", type=int, 
                        default=int(os.getenv('CHUNK_SIZE', DEFAULT_CHUNK_SIZE)),
                        help="Maximum chunk size in characters")
    parser.add_argument("--buffer_size", type=int, 
                        default=int(os.getenv('BUFFER_SIZE', 2)),
                        help="Buffer size for semantic splitter")
    parser.add_argument("--breakpoint_threshold", type=int, 
                        default=int(os.getenv('BREAKPOINT_THRESHOLD', 90)),
                        help="Breakpoint percentile threshold")
    
    # Embedding parameters
    parser.add_argument("--embedding_model", type=str, 
                        default=os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL),
                        help="Embedding model to use")
    parser.add_argument("--batch_size", type=int, 
                        default=int(os.getenv('BATCH_SIZE', 50)),
                        help="Batch size for uploading to Pinecone")
    
    return parser.parse_args()

def find_pdf_files(directory: str) -> List[Path]:
    """Recursively find all PDF files in directory."""
    pdf_dir = Path(directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    return pdf_files

def load_existing_metadata(metadata_file: str) -> Dict[str, Dict]:
    """Load existing metadata from CSV scraping results."""
    if not metadata_file or not os.path.exists(metadata_file):
        return {}
    
    import pandas as pd
    try:
        df = pd.read_csv(metadata_file)
        metadata_dict = {}
        for _, row in df.iterrows():
            filename = row.get('filename', '')
            if filename:
                metadata_dict[filename] = row.to_dict()
        logger.info(f"Loaded metadata for {len(metadata_dict)} files")
        return metadata_dict
    except Exception as e:
        logger.warning(f"Could not load metadata file: {e}")
        return {}

def process_pdf_files(pdf_files: List[Path], existing_metadata: Dict) -> List[Document]:
    """Process PDF files into LlamaIndex documents."""
    processor = PDFProcessor()
    documents = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Extract content and metadata
            text, metadata = processor.extract_pdf_content(pdf_path)
            
            # Merge with existing metadata if available
            filename_base = pdf_path.stem
            if filename_base in existing_metadata:
                scraped_metadata = existing_metadata[filename_base]
                metadata.update({
                    'scraped_title': scraped_metadata.get('title', ''),
                    'scraped_authors': scraped_metadata.get('authors', ''),
                    'scraped_url': scraped_metadata.get('url', ''),
                    'scraped_date': scraped_metadata.get('publish_date', ''),
                    'scraped_type': scraped_metadata.get('publication_type', ''),
                    'scraped_doi': scraped_metadata.get('doi', '')
                })
            
            # Create document
            doc = Document(
                text=text,
                metadata=metadata,
                doc_id=f"cdc_tbi_{len(documents)}"
            )
            
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(documents)} PDF documents")
    return documents

def setup_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[BaseEmbedding, int]:
    """Set up and return the embedding model."""
    if model_name == "BAAI/bge-small-en-v1.5":
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
        # Default to bge-small
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            embed_batch_size=32
        )
        dimension = 384
        
    logger.info(f"Using embedding model: {model_name} with dimension {dimension}")
    return embed_model, dimension

def initialize_pinecone(api_key: str, index_name: str, dimension: int):
    """Initialize Pinecone index."""
    try:        
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_exists = any(idx.name == index_name for idx in existing_indexes)
        
        if not index_exists:
            logger.info(f"Creating new Pinecone index: {index_name}")
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
        
        index = pc.Index(index_name)
        return index
    
    except Exception as e:
        logger.error(f"Pinecone initialization error: {e}")
        raise

def chunk_documents(documents: List[Document], embed_model: BaseEmbedding, 
                   chunk_size: int, buffer_size: int, breakpoint_threshold: int) -> List:
    """Chunk documents using TBI-optimized splitter or basic chunking."""
    
    if TBIOptimizedSemanticSplitter:
        logger.info("Using TBI-Optimized Semantic Splitter")
        splitter = TBIOptimizedSemanticSplitter(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_threshold,
            embedding_length=chunk_size,
            importance_weight=0.4
        )
        
        all_nodes = []
        for doc in tqdm(documents, desc="Chunking documents"):
            nodes = splitter.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
    else:
        # Basic chunking fallback
        logger.info("Using basic text chunking")
        from llama_index.core.text_splitter import SentenceSplitter
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
        
        all_nodes = []
        for doc in tqdm(documents, desc="Chunking documents"):
            nodes = splitter.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
    
    logger.info(f"Created {len(all_nodes)} chunks from {len(documents)} documents")
    return all_nodes

def upload_to_pinecone(index, nodes, batch_size=50, output_dir="."):
    """Upload nodes to Pinecone with progress tracking."""
    total_nodes = len(nodes)
    batches = [nodes[i:i+batch_size] for i in range(0, total_nodes, batch_size)]
    
    logger.info(f"Uploading {total_nodes} nodes to Pinecone in {len(batches)} batches")
    
    upload_summary = {
        "total_chunks": total_nodes,
        "total_batches": len(batches),
        "successful_uploads": 0,
        "failed_uploads": 0,
        "start_time": time.time()
    }
    
    for i, batch in enumerate(tqdm(batches, desc="Uploading to Pinecone")):
        try:
            vectors = []
            
            for node in batch:
                # Ensure embedding exists
                if node.embedding is None:
                    logger.warning(f"Node {node.node_id} has no embedding, skipping")
                    continue
                
                # Prepare metadata for Pinecone
                metadata = dict(node.metadata)
                
                # Add text content (truncated for Pinecone limits)
                metadata["text"] = node.text[:1000] if len(node.text) > 1000 else node.text
                
                # Ensure all values are strings/numbers
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ", ".join(str(v) for v in value)
                    elif not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                
                vectors.append({
                    "id": node.node_id,
                    "values": node.embedding,
                    "metadata": metadata
                })
            
            # Upload batch
            if vectors:
                index.upsert(vectors=vectors)
                upload_summary["successful_uploads"] += len(vectors)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Failed to upload batch {i}: {e}")
            upload_summary["failed_uploads"] += len(batch)
    
    upload_summary["end_time"] = time.time()
    upload_summary["duration"] = upload_summary["end_time"] - upload_summary["start_time"]
    
    # Save upload summary
    with open(os.path.join(output_dir, "upload_summary.json"), 'w') as f:
        json.dump(upload_summary, f, indent=2)
    
    logger.info(f"Upload completed: {upload_summary['successful_uploads']} successful, "
                f"{upload_summary['failed_uploads']} failed")
    
    return upload_summary

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to output directory
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "pdf_ingestion.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting CDC TBI PDF ingestion pipeline")
    logger.info(f"PDF Directory: {args.pdf_directory}")
    logger.info(f"Target Index: {args.index_name}")
    
    # Find PDF files
    pdf_files = find_pdf_files(args.pdf_directory)
    
    # Load existing metadata if available
    existing_metadata = load_existing_metadata(args.metadata_file)
    
    # Process PDFs into documents
    documents = process_pdf_files(pdf_files, existing_metadata)
    
    if not documents:
        logger.error("No documents were successfully processed. Exiting.")
        return
    
    # Setup embedding model
    embed_model, dimension = setup_embedding_model(args.embedding_model)
    
    # Initialize Pinecone
    index = initialize_pinecone(args.pinecone_api_key, args.index_name, dimension)
    
    # Chunk documents
    nodes = chunk_documents(
        documents, embed_model, args.chunk_size, 
        args.buffer_size, args.breakpoint_threshold
    )
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    for node in tqdm(nodes, desc="Embedding chunks"):
        if node.embedding is None:
            embedding = embed_model.get_text_embedding(node.text)
            node.embedding = embedding
    
    # Save processing results
    processing_summary = {
        "total_pdfs_found": len(pdf_files),
        "successfully_processed": len(documents),
        "total_chunks_created": len(nodes),
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "processing_date": datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, "processing_summary.json"), 'w') as f:
        json.dump(processing_summary, f, indent=2)
    
    # Upload to Pinecone
    upload_summary = upload_to_pinecone(
        index, nodes, args.batch_size, args.output_dir
    )
    
    logger.info(f"CDC TBI PDF ingestion completed!")
    logger.info(f"Processed {len(documents)} PDFs into {len(nodes)} chunks")
    logger.info(f"Successfully uploaded {upload_summary['successful_uploads']} chunks to Pinecone")

if __name__ == "__main__":
    main()