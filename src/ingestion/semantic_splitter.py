from typing import Any, Callable, List, Optional, Sequence, TypedDict, Dict, Set

import numpy as np
import re
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, Document, MetadataMode
from llama_index.core.utils import get_tqdm_iterable

from textwrap import wrap

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

# Medical domain-specific keywords that often indicate important content
TBI_KEYWORDS = {
    "traumatic brain injury", "tbi", "concussion", "brain injury", "head trauma",
    "post-concussion", "neurological", "cognitive", "balance problems", "memory loss",
    "headache", "dizziness", "nausea", "sensitivity to light", "sensitivity to noise",
    "vestibular", "rehabilitation", "recovery", "symptoms", "treatment", "diagnosis",
    "chronic traumatic encephalopathy", "cte", "impact injury", "iiss", "severity score",
    "neuroplasticity", "neurology", "stroke", "patient", "impact"
}

class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: List[float]
    importance_score: float


class TBIOptimizedSemanticSplitter(NodeParser):
    """TBI-Optimized Semantic Splitter for Power of Patients articles.
    
    Specialized for medical content related to traumatic brain injuries,
    concussions, and neurological issues. Recognizes domain-specific
    terminology and optimizes chunking for medical context.

    Args:
        buffer_size (int): number of sentences to group together when evaluating semantic similarity
        embed_model: (BaseEmbedding): embedding model to use
        sentence_splitter (Optional[Callable]): splits text into sentences
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
        medical_keywords (Set[str]): domain-specific keywords to identify important content
        importance_weight (float): weight for the importance score (0-1)
    """

    embedding_length: int = Field(
        default=512,
        description=(
            "The max embedding length for the model."
        ),
    )

    sentence_splitter: Callable[[str], List[str]] = Field(
        default_factory=split_by_sentence_tokenizer,
        description="The text splitter to use when splitting documents.",
        exclude=True,
    )

    embed_model: BaseEmbedding = Field(
        description="The embedding model to use for semantic comparison",
    )

    buffer_size: int = Field(
        default=2,  # Increased from 1 to better capture medical context
        description=(
            "The number of sentences to group together when evaluating semantic similarity. "
            "Set to 2+ to capture medical concepts that often span multiple sentences."
        ),
    )

    breakpoint_percentile_threshold: int = Field(
        default=90,  # Decreased from 95 to create smaller, more focused chunks
        description=(
            "The percentile of cosine dissimilarity that must be exceeded between a "
            "group of sentences and the next to form a node. Lower values create more chunks."
        ),
    )
    
    medical_keywords: Set[str] = Field(
        default_factory=lambda: TBI_KEYWORDS,
        description="Domain-specific medical keywords related to TBI and neurological issues",
    )
    
    importance_weight: float = Field(
        default=0.3,
        description="Weight given to keyword importance when determining chunk boundaries (0-1)",
    )
    
    min_chunk_size: int = Field(
        default=50,  # Minimum characters in a chunk
        description="Minimum size for a chunk to be considered valid",
    )
    
    max_chunk_size: int = Field(
        default=1024,  # Maximum characters in a chunk
        description="Maximum size for a chunk before forcing a split",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TBIOptimizedSemanticSplitter"

    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: Optional[int] = 90,
        buffer_size: Optional[int] = 2,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
        medical_keywords: Optional[Set[str]] = None,
        importance_weight: Optional[float] = 0.3,
        min_chunk_size: Optional[int] = 50,
        max_chunk_size: Optional[int] = 1024,
    ) -> "TBIOptimizedSemanticSplitter":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        if embed_model is None:
            try:
                from llama_index.embeddings.openai import (
                    OpenAIEmbedding,
                )  # pants: no-infer-dep

                embed_model = embed_model or OpenAIEmbedding()
            except ImportError:
                try:
                    from llama_index.embeddings.huggingface import (
                        HuggingFaceEmbedding,
                    )
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                except ImportError:
                    raise ImportError(
                        "No embedding model found. Please install llama-index-embeddings-openai"
                        "or llama-index-embeddings-huggingface"
                    )

        id_func = id_func or default_id_func
        medical_keywords = medical_keywords or TBI_KEYWORDS

        return cls(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
            medical_keywords=medical_keywords,
            importance_weight=importance_weight,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes into semantically meaningful chunks."""
        # Set embedding length from kwargs if provided
        self.embedding_length = kwargs.get("embedding_length", self.embedding_length)
        
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            # Extract article metadata if available
            metadata = {}
            if hasattr(node, "metadata"):
                metadata = node.metadata or {}
            
            # Convert node to Document with metadata
            if isinstance(node, Document):
                doc = node
            else:
                # Create Document from node
                doc = Document(
                    text=node.get_content(metadata_mode=MetadataMode.NONE),
                    metadata=metadata,
                )
            
            # Generate semantic nodes
            semantic_nodes = self.build_semantic_nodes_from_documents([doc], show_progress)
            all_nodes.extend(semantic_nodes)

        return all_nodes

    def build_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Build semantic nodes from documents optimized for TBI medical content."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            text = doc.text
            
            # Pre-process text to handle medical abbreviations and terminology
            text = self._preprocess_medical_text(text)
            
            # Skip empty documents
            if not text.strip():
                continue
                
            # Split text into sentences
            text_splits = self.sentence_splitter(text)
            
            # Skip if no sentences
            if not text_splits:
                continue

            # Build sentence groups with importance scoring
            sentences = self._build_sentence_groups_with_importance(text_splits)

            # Handle sentences that exceed embedding length
            new_sentences = []
            for sentence in sentences:
                if len(sentence["combined_sentence"]) >= self.embedding_length:
                    # Split long sentences while preserving context
                    splitted_sentences = self._split_long_sentence(sentence)
                    new_sentences.extend(splitted_sentences)
                else:
                    new_sentences.append(sentence)

            # Skip if no sentences
            if not new_sentences:
                continue
                
            # Get embeddings for all sentences
            combined_sentence_embeddings = self.embed_model.get_text_embedding_batch(
                [x["combined_sentence"] for x in new_sentences], 
                show_progress=show_progress,
            )

            # Assign embeddings to sentences
            for i, embedding in enumerate(combined_sentence_embeddings):
                new_sentences[i]["combined_sentence_embedding"] = embedding

            # Calculate semantic distances between adjacent sentences
            distances = self._calculate_distances_between_sentence_groups(new_sentences)
            
            # Skip if no distances
            if not distances:
                # If only one sentence, create a single node
                if len(new_sentences) == 1:
                    node = build_nodes_from_splits(
                        [new_sentences[0]["combined_sentence"]], 
                        doc,
                        id_func=self.id_func,
                    )[0]
                    all_nodes.append(node)
                continue
            
            # Adjust distances based on medical importance
            adjusted_distances = self._adjust_distances_by_importance(new_sentences, distances)

            # Build chunks based on semantic boundaries
            chunks = self._build_node_chunks(new_sentences, adjusted_distances)
            
            # Skip if no chunks
            if not chunks:
                continue

            # Create nodes from chunks
            nodes = build_nodes_from_splits(
                chunks,
                doc,
                id_func=self.id_func,
            )

            all_nodes.extend(nodes)

        return all_nodes
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text to handle abbreviations and formatting."""
        if not text:
            return ""
            
        # Ensure proper spacing after periods in abbreviations
        text = re.sub(r'([A-Z])\.([A-Z])', r'\1. \2', text)
        
        # Fix common medical abbreviation patterns
        text = re.sub(r'TBI\.', r'TBI. ', text)
        text = re.sub(r'e\.g\.', r'e.g. ', text)
        text = re.sub(r'i\.e\.', r'i.e. ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_keyword_importance(self, text: str) -> float:
        """Calculate importance score based on medical keyword presence."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        
        # Count keyword matches
        keyword_count = sum(1 for keyword in self.medical_keywords if keyword in text_lower)
        
        # Normalize by text length and apply sigmoid-like scaling
        # This ensures the score is between 0 and 1
        text_length = max(1, len(text) / 100)  # Normalize by 100 chars
        raw_score = keyword_count / text_length
        
        # Apply sigmoid-like function to keep scores between 0 and 1
        importance = min(1.0, raw_score * 2)
        
        return importance

    def _build_sentence_groups_with_importance(
        self, text_splits: List[str]
    ) -> List[SentenceCombination]:
        """Build sentence groups with importance scoring for medical content."""
        if not text_splits:
            return []
            
        sentences: List[SentenceCombination] = [
            {
                "sentence": x,
                "index": i,
                "combined_sentence": "",
                "combined_sentence_embedding": [],
                "importance_score": 0.0,
            }
            for i, x in enumerate(text_splits)
        ]

        # Group sentences and calculate embeddings for sentence groups
        for i in range(len(sentences)):
            combined_sentence = ""

            # Include preceding sentences based on buffer size
            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]

            # Add current sentence
            combined_sentence += sentences[i]["sentence"]

            # Include following sentences based on buffer size
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence
            
            # Calculate importance score based on medical keywords
            sentences[i]["importance_score"] = self._calculate_keyword_importance(combined_sentence)

        return sentences
    
    def _split_long_sentence(self, sentence: SentenceCombination) -> List[SentenceCombination]:
        """Split long sentences while preserving context and medical terminology."""
        original_sentence = sentence["sentence"]
        combined_sentence = sentence["combined_sentence"]
        
        if not combined_sentence:
            return []
        
        # Use textwrap.wrap but preserve medical terms
        # First identify medical terms we want to keep together
        medical_terms_pattern = '|'.join(re.escape(term) for term in self.medical_keywords if len(term.split()) > 1)
        
        # If we have medical terms to preserve
        if medical_terms_pattern:
            # Replace spaces in medical terms with temporary marker
            for term in self.medical_keywords:
                if ' ' in term:
                    term_pattern = re.escape(term)
                    combined_sentence = re.sub(
                        f"(?i){term_pattern}", 
                        lambda m: m.group(0).replace(' ', '___MEDICAL_TERM_SPACE___'), 
                        combined_sentence
                    )
        
        # Now split the text
        splits = wrap(combined_sentence, self.embedding_length)
        
        # Restore spaces in medical terms
        if medical_terms_pattern:
            splits = [split.replace('___MEDICAL_TERM_SPACE___', ' ') for split in splits]
        
        # Create new sentence combinations
        result = []
        for i, split_text in enumerate(splits):
            result.append({
                "sentence": original_sentence if i == 0 else "",  # Only include original sentence in first split
                "index": i,
                "combined_sentence": split_text,
                "combined_sentence_embedding": [],
                "importance_score": sentence["importance_score"],
            })
        
        return result

    def _calculate_distances_between_sentence_groups(
        self, sentences: List[SentenceCombination]
    ) -> List[float]:
        """Calculate semantic distances between adjacent sentence groups."""
        distances = []
        
        if len(sentences) <= 1:
            return distances
            
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]

            # Handle case where embeddings might be empty
            if not embedding_current or not embedding_next:
                # Default to medium distance if embeddings are missing
                distances.append(0.5)
                continue

            try:
                similarity = self.embed_model.similarity(embedding_current, embedding_next)
                distance = 1 - similarity
                distances.append(distance)
            except Exception as e:
                # Default to medium distance if similarity calculation fails
                distances.append(0.5)

        return distances
    
    def _adjust_distances_by_importance(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[float]:
        """Adjust distances based on medical importance scores."""
        if not distances:
            return distances
            
        adjusted_distances = distances.copy()
        
        for i in range(len(distances)):
            # Get importance scores for adjacent sentences
            current_importance = sentences[i]["importance_score"]
            next_importance = sentences[i+1]["importance_score"] if i+1 < len(sentences) else 0
            
            # Calculate importance difference - higher difference suggests potential topic change
            importance_diff = abs(current_importance - next_importance)
            
            # Boost distances where there's a change in importance or high importance
            importance_factor = importance_diff * self.importance_weight
            
            # Additional boost if both sentences have high medical relevance
            if current_importance > 0.5 and next_importance > 0.5:
                # If both are important, we want to keep them together (decrease distance)
                importance_factor -= 0.1
            
            # Apply the adjustment
            adjusted_distances[i] = min(1.0, max(0.0, distances[i] + importance_factor))
            
        return adjusted_distances

    def _build_node_chunks(
        self, sentences: List[SentenceCombination], distances: List[float]
    ) -> List[str]:
        """Build node chunks optimized for TBI medical content."""
        chunks = []
        
        if len(distances) > 0:
            # Calculate threshold based on percentile of distances
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            # Find indices where distance exceeds threshold (potential topic changes)
            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                
                # Combine sentences into a single text chunk
                combined_text = "".join([d["sentence"] for d in group])
                
                # Check if combined text exceeds maximum chunk size
                if len(combined_text) > self.max_chunk_size:
                    # Split into smaller chunks while preserving sentence boundaries
                    sub_chunks = []
                    current_chunk = ""
                    
                    for sent in group:
                        if len(current_chunk) + len(sent["sentence"]) > self.max_chunk_size:
                            if len(current_chunk) >= self.min_chunk_size:
                                sub_chunks.append(current_chunk)
                            current_chunk = sent["sentence"]
                        else:
                            current_chunk += sent["sentence"]
                    
                    if current_chunk and len(current_chunk) >= self.min_chunk_size:
                        sub_chunks.append(current_chunk)
                    
                    chunks.extend(sub_chunks)
                else:
                    # Add combined text as a chunk if it meets minimum size
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)

                start_index = index + 1

            # Process remaining sentences
            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                
                # Check if combined text exceeds maximum chunk size
                if len(combined_text) > self.max_chunk_size:
                    # Split into smaller chunks
                    text_chunks = wrap(combined_text, self.max_chunk_size)
                    # Filter out chunks that are too small
                    chunks.extend([c for c in text_chunks if len(c) >= self.min_chunk_size])
                else:
                    # Add as a single chunk if it meets minimum size
                    if len(combined_text) >= self.min_chunk_size:
                        chunks.append(combined_text)
        else:
            # If there are no distances (very small document), treat as single node
            combined_text = " ".join([s["sentence"] for s in sentences])
            
            # Split if necessary
            if len(combined_text) > self.max_chunk_size:
                text_chunks = wrap(combined_text, self.max_chunk_size)
                chunks.extend([c for c in text_chunks if len(c) >= self.min_chunk_size])
            else:
                if len(combined_text) >= self.min_chunk_size:
                    chunks.append(combined_text)

        return chunks
    
