"""Vector store for document embeddings."""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional
import faiss
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        
        # Initialize index if needed
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, query_text: str, 
              k: int = 3, use_hybrid: bool = False, alpha: float = 0.7,
              rerank: bool = False) -> List[Dict[str, Any]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Embedding of the query.
            query_text: Original query text (not used in basic search).
            k: Number of results to return.
            use_hybrid: Not used in basic implementation.
            alpha: Not used in basic implementation.
            rerank: Not used in basic implementation.
            
        Returns:
            List of dictionaries with search results.
        """
        if self.index is None or not self.texts:
            return []
            
        # Use vector search
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        indices = indices[0]
        distances = distances[0]
        
        # Format results
        results = []
        for i, idx in enumerate(indices):
            if idx >= len(self.texts):  # Safety check
                continue
                
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(1 / (1 + distances[i]))  # Convert distance to similarity score
            })
        
        return results
    
    def search_with_filter(self, query_embedding: np.ndarray, query_text: str, 
                          filter_key: str, filter_value: Any, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents with a metadata filter.
        
        Args:
            query_embedding: Embedding of the query.
            query_text: Original query text.
            filter_key: Metadata key to filter on.
            filter_value: Value to match for the filter key.
            k: Number of results to return.
            
        Returns:
            List of dictionaries with filtered search results.
        """
        # First get more results than needed
        initial_results = self.search(query_embedding, query_text, k=k*3)
        
        # Filter by metadata
        filtered_results = [r for r in initial_results if r["metadata"].get(filter_key) == filter_value]
        
        # Return top k after filtering
        return filtered_results[:k]
    
    def get_document_by_id(self, doc_id: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its ID in metadata.
        
        Args:
            doc_id: ID to search for.
            
        Returns:
            Dictionary with document info or None if not found.
        """
        for i, meta in enumerate(self.metadata):
            if meta.get("chunk_id") == doc_id or meta.get("id") == doc_id:
                return {
                    "text": self.texts[i],
                    "metadata": meta
                }
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
        
        return [{"text": d["text"], "metadata": d["metadata"]} for d in source_docs[start:end]]
    
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
            "version": "1.0"
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
            "sources": sources
        }
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = None
        self.texts = []
        self.metadata = []
        logger.info("Vector store cleared")