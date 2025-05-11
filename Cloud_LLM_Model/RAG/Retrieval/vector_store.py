"""Vector store for document embeddings with improvements."""
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from cachetools import LRUCache
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from functools import lru_cache
import hashlib

# Cache for storing embeddings - minimize API calls
embedding_cache = LRUCache(maxsize=1000)  # Increased cache size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStore")


def _make_hashable(value):
    """Convert unhashable types to hashable ones."""
    if isinstance(value, list):
        return tuple(_make_hashable(v) for v in value)
    elif isinstance(value, set):
        return tuple(sorted(_make_hashable(v) for v in value))
    elif isinstance(value, np.ndarray):
        return tuple(value.flatten().tolist())
    elif isinstance(value, dict):
        return {k: _make_hashable(v) for k, v in value.items()}
    else:
        return value


def get_embedding_hash(embedding: Union[np.ndarray, List[float], tuple]) -> str:
    """
    Generate a consistent hash for any embedding format.
    
    Args:
        embedding: The embedding in any format (list, tuple, or numpy array)
        
    Returns:
        SHA-256 hash string
    """
    # Convert to numpy array if needed
    if isinstance(embedding, (list, tuple)):
        arr = np.array(embedding, dtype=np.float32)
    elif isinstance(embedding, np.ndarray):
        arr = embedding.astype(np.float32)
    else:
        raise TypeError(f"embedding must be list, tuple, or numpy array, got {type(embedding)}")
    
    # Generate hash from the bytes representation
    return hashlib.sha256(arr.tobytes()).hexdigest()


class Vector_Store:
    """Class for storing and retrieving document embeddings."""
    
    def __init__(self, dimension: int = 3072):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimension of vectors to store.
        """
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.metadata = []
        # Keep track of normalized vectors
        self.is_normalized = False

    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> None:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of text documents.
            embeddings: NumPy array of embeddings.
            metadata: Optional list of metadata dictionaries for each text.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
        
        if metadata is None:
            metadata = [{"index": i} for i in range(len(texts))]
        elif len(metadata) != len(texts):
            raise ValueError("Number of metadata items must match number of texts")
        
        # Sanitize metadata before storing
        sanitized_metadata = []
        for meta in metadata:
            sanitized_meta = {}
            for key, value in meta.items():
                sanitized_meta[key] = _make_hashable(value)
            sanitized_metadata.append(sanitized_meta)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = normalize(embeddings.astype(np.float32), axis=1)
        self.is_normalized = True
        
        # Initialize index if needed
        if self.index is None:
            self.dimension = normalized_embeddings.shape[1]
            # Use IndexFlatIP for cosine similarity with normalized vectors
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Store texts and sanitized metadata
        self.texts.extend(texts)
        self.metadata.extend(sanitized_metadata)
        
        # Add to FAISS index
        self.index.add(normalized_embeddings)
        
        logger.info(f"Added {len(texts)} documents to vector store")

    def search(self, query_embedding: Union[np.ndarray, List[float], tuple], query_text: str, 
               k: int = 3, use_hybrid: bool = True, alpha: float = 0.7,
               rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Search for most similar documents with optional hybrid retrieval and reranking.
        
        Args:
            query_embedding: Embedding of the query (can be list, tuple, or numpy array).
            query_text: Original query text (used for hybrid search).
            k: Number of results to return.
            use_hybrid: Whether to use hybrid retrieval (vector + sparse).
            alpha: Weight for vector similarity in hybrid retrieval.
            rerank: Whether to rerank results using a simple relevance score.
            
        Returns:
            List of dictionaries with search results.
        """
        if self.index is None or not self.texts:
            return []
            
        # Cache key for this query - using helper function to handle all formats
        embedding_hash = get_embedding_hash(query_embedding)
        cache_key = f"{embedding_hash}:{query_text}:{k}:{use_hybrid}:{alpha}:{rerank}"
        
        # Check if we have cached results for this query
        if cache_key in embedding_cache:
            logger.info("Using cached search results")
            return embedding_cache[cache_key]
        
        # Convert to numpy array if needed
        if isinstance(query_embedding, (list, tuple)):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        elif not isinstance(query_embedding, np.ndarray):
            raise TypeError(f"query_embedding must be list, tuple, or numpy array, got {type(query_embedding)}")
            
        # Normalize query embedding for cosine similarity
        normalized_query = normalize(query_embedding.reshape(1, -1).astype(np.float32))
        
        # Use vector search (now using cosine similarity via dot product)
        distances, indices = self.index.search(normalized_query, k*2)  # Get more results for reranking
        indices = indices[0]
        distances = distances[0]
        
        # Format results
        results = []
        for i, idx in enumerate(indices):
            if idx >= len(self.texts) or idx < 0:  # Safety check
                continue
                
            # Convert distance to similarity score (IP distances are already similarities)
            similarity = float(distances[i])
            
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": similarity
            })
        
        # Rerank results if requested
        if rerank and results:
            results = self._rerank_results(results, query_text)
            
        # Return top-k after reranking
        final_results = results[:k]
        
        # Cache the results
        embedding_cache[cache_key] = final_results
        
        return final_results

    def search_with_filter(self, query_embedding: Union[np.ndarray, List[float], tuple], query_text: str, 
                        filter_key: str, filter_value: Any, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents with a metadata filter.
        
        Args:
            query_embedding: Embedding of the query (can be list, tuple, or numpy array).
            query_text: Original query text.
            filter_key: Metadata key to filter on.
            filter_value: Value to match for the filter key.
            k: Number of results to return.
            
        Returns:
            List of dictionaries with filtered search results.
        """
        # Create cache key for filtered search - using helper function
        embedding_hash = get_embedding_hash(query_embedding)
        
        # Handle unhashable filter values
        try:
            filter_value_str = json.dumps(filter_value, sort_keys=True)
        except TypeError:
            filter_value_str = str(filter_value)
        
        cache_key = f"{embedding_hash}:{query_text}:{filter_key}:{filter_value_str}:{k}"
        
        # Check cache first
        if cache_key in embedding_cache:
            logger.info("Using cached filtered search results")
            return embedding_cache[cache_key]
        
        # First get more results than needed
        initial_results = self.search(query_embedding, query_text, k=k*3)
        
        # Filter by metadata
        filtered_results = [r for r in initial_results if r["metadata"].get(filter_key) == filter_value]
        
        # Return top k after filtering
        result = filtered_results[:k]
        embedding_cache[cache_key] = result
        
        return result

    def get_document_by_id(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its ID in metadata.
        
        Args:
            doc_id: ID to search for.
            
        Returns:
            Dictionary with document info or None if not found.
        """
        # Create cache key
        cache_key = f"doc_id:{doc_id}"
        
        # Check cache
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]
            
        for i, meta in enumerate(self.metadata):
            if meta.get("chunk_id") == doc_id or meta.get("id") == doc_id:
                result = {
                    "text": self.texts[i],
                    "metadata": meta
                }
                # Cache result
                embedding_cache[cache_key] = result
                return result
        return None

    def get_context_window(self, doc_id: Any, window_size: int = 1) -> List[Dict[str, Any]]:
        """
        Get surrounding context for a specific document.
        
        Args:
            doc_id: ID of the center document.
            window_size: Number of documents to include on each side.
            
        Returns:
            List of dictionaries with context documents.
        """
        # Create cache key
        cache_key = f"context:{doc_id}:{window_size}"
        
        # Check cache
        if cache_key in embedding_cache:
            return embedding_cache[cache_key]
            
        # Find the document first
        doc = self.get_document_by_id(doc_id)
        if not doc:
            return []
        
        source = doc["metadata"].get("source")
        
        # Find all documents from the same source
        source_docs = []
        for i, meta in enumerate(self.metadata):
            if meta.get("source") == source:
                source_docs.append({
                    "index": i,
                    "chunk_id": meta.get("chunk_id", meta.get("id")),
                    "text": self.texts[i],
                    "metadata": meta
                })
        
        # Sort by chunk_id or sequence
        source_docs.sort(key=lambda x: x["chunk_id"])
        
        # Find the current document position
        current_pos = next((i for i, d in enumerate(source_docs) 
                            if d["chunk_id"] == doc["metadata"].get("chunk_id", doc["metadata"].get("id"))), -1)
        
        if current_pos == -1:
            return []
        
        # Get the window
        start = max(0, current_pos - window_size)
        end = min(len(source_docs), current_pos + window_size + 1)
        
        result = [{"text": d["text"], "metadata": d["metadata"]} for d in source_docs[start:end]]
        
        # Cache result
        embedding_cache[cache_key] = result
        
        return result

    def _rerank_results(self, results: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """
        Improved reranking of results based on text similarity and vector score.
        
        Args:
            results: Initial search results.
            query_text: Original query text.
            
        Returns:
            Reranked results.
        """
        # Convert query to lowercase for matching
        query_lower = query_text.lower()
        query_terms = set(query_lower.split())
        
        for result in results:
            text = result["text"].lower()
            
            # Calculate term overlap with tf-idf like approach
            text_terms = set(text.split())
            common_terms = query_terms.intersection(text_terms)
            
            # Weight rare terms higher
            term_weights = {}
            for term in common_terms:
                # Count occurrences of term in all documents (simplified TF-IDF)
                doc_count = sum(1 for t in self.texts if term in t.lower())
                # Avoid division by zero
                doc_count = max(1, doc_count)
                # Inverse document frequency
                idf = np.log(len(self.texts) / doc_count)
                term_weights[term] = idf
            
            # Weighted term overlap
            term_score = sum(term_weights.values()) / max(1, sum(1 for _ in query_terms))
            
            # Exact phrase matching with context
            phrase_bonus = 0.0
            if query_lower in text:
                phrase_bonus = 0.3  # Increased bonus for exact matches
                
                # Get context around the matched phrase
                phrase_idx = text.find(query_lower)
                context_start = max(0, phrase_idx - 50)
                context_end = min(len(text), phrase_idx + len(query_lower) + 50)
                context = text[context_start:context_end]
                
                # Bonus for phrases in important positions (beginning of text)
                if phrase_idx < 100:
                    phrase_bonus += 0.1
            
            # Context relevance (presence of key entities)
            # More sophisticated NLP would be better, but this is a simple approach
            context_score = 0.0
            for term in query_terms:
                if len(term) > 3:  # Only consider meaningful terms
                    if text.count(term) > 1:  # Term appears multiple times
                        context_score += 0.05
            
            # Calculate chunk length penalty (prefer shorter, more focused chunks)
            # But also consider too short chunks as potentially incomplete information
            optimal_length = 500  # Optimal chunk size
            length_factor = min(1.0, optimal_length / max(1, len(text)))
            length_penalty = 1.0 - abs(len(text) - optimal_length) / 1000  # Penalize far from optimal
            length_penalty = max(0.5, min(1.0, length_penalty))  # Keep between 0.5 and 1.0
            
            # Metadata relevance
            metadata_score = 0.0
            if "source" in result["metadata"]:
                source = result["metadata"]["source"]
                # Prefer sources with names related to the query
                for term in query_terms:
                    if term in str(source).lower():
                        metadata_score += 0.1
            
            # Final reranking score combines vector similarity with text-based signals
            result["rerank_score"] = (
                result["score"] * 0.5 +          # Vector similarity
                term_score * 0.2 +               # Term overlap with weights
                phrase_bonus +                   # Exact phrase matching
                context_score * 0.1 +            # Context relevance
                length_penalty * 0.1 +           # Length considerations
                metadata_score                   # Metadata relevance
            )
        
        # Sort by reranking score
        results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return results

    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store files.
        """
        if self.index is None:
            logger.warning("No index to save. Please add documents first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save config
        config = {
            "dimension": self.dimension,
            "index_size": len(self.texts),
            "version": "1.0",
            "is_normalized": self.is_normalized
        }
        
        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(config, f)
        
        # Save texts and metadata
        with open(os.path.join(directory, "texts.pkl"), 'wb') as f:
            pickle.dump(self.texts, f)
        
        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f)
        
        logger.info(f"Vector store saved to {directory}")

    def load(self, directory: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory containing the vector store files.
            
        Returns:
            True if successful, False otherwise.
        """
        index_path = os.path.join(directory, "faiss_index.bin")
        texts_path = os.path.join(directory, "texts.pkl")
        metadata_path = os.path.join(directory, "metadata.json")
        config_path = os.path.join(directory, "config.json")
        
        if not (os.path.exists(index_path) and os.path.exists(texts_path) and os.path.exists(metadata_path)):
            logger.error(f"Missing files in {directory}")
            return False
        
        # Load config if available
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.dimension = config.get("dimension", 1536)
                self.is_normalized = config.get("is_normalized", False)
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load texts
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Vector store loaded from {directory} with {len(self.texts)} documents")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics.
        """
        if self.index is None:
            return {
                "status": "empty",
                "document_count": 0
            }
        
        # Count documents by source
        sources = {}
        for meta in self.metadata:
            source = meta.get("source", "unknown")
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        return {
            "status": "loaded",
            "document_count": len(self.texts),
            "dimension": self.dimension,
            "sources": sources,
            "is_normalized": self.is_normalized
        }

    def clear(self) -> None:
        """Clear the vector store."""
        self.index = None
        self.texts = []
        self.metadata = []
        # Clear cache too
        embedding_cache.clear()
        logger.info("Vector store cleared")