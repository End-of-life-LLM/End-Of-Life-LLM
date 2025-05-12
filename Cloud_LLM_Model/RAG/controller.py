"""Main controller for Retrieval Augmented Generation with improvements"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
import hashlib

# Import from our modules
from .embedding_service import Embedding_Service
from Cloud_LLM_Model.Utils.token_counter import Token_Counter
from Cloud_LLM_Model.RAG.Retrieval.vector_store import Vector_Store
from Cloud_LLM_Model.RAG.Text_Utils.evaluation_utils import Evaluation_Utils
from Cloud_LLM_Model.RAG.Text_Utils.text_processor import Text_Processor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGController:
    """
    Controller class that combines embedding generation and retrieval functionality
    for a complete RAG (Retrieval Augmented Generation) system.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                embedding_model: str = "text-embedding-3-large",
                tier: str = "free",
                cache_enabled: bool = True):
        """
        Initialize the RAG controller.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            embedding_model: Name of the embedding model to use.
            tier: API tier for rate limiting ('free' or 'tier1').
            cache_enabled: Whether to enable caching for embeddings and search results.
        """
        # Initialize embedding service
        self.embedding_service = Embedding_Service(
            api_key=api_key,
            embedding_model=embedding_model,
            tier=tier,
            cache_enabled=cache_enabled
        )
        
        # Initialize vector store with the same dimension as embedding model
        self.vector_store = Vector_Store(dimension=self.embedding_service.dimension)
        
        # Keep a token counter instance for convenience
        self.token_counter = Token_Counter()
        
        # Create text processor instance
        self.text_processor = Text_Processor()
        
        # Cache for search results
        self.query_cache = {}
        self.cache_enabled = cache_enabled
    
    # ------------------ Indexing Functions ------------------
    
    def index_text_file(self, file_path: str, chunk_size: int = None, chunk_overlap: int = 200) -> int:
        """
        Index a text file by splitting, embedding, and storing in vector store.
        Now with automatic chunk size determination.
        
        Args:
            file_path: Path to the text file.
            chunk_size: Maximum size of each chunk, or None for auto-determination.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            Number of chunks indexed.
        """
        # Chunk the text file - now with dynamic chunk sizing if chunk_size is None
        chunks, metadata = self.text_processor.read_and_chunk_file(
            file_path, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Add token counts to metadata
        for i, chunk in enumerate(metadata):
            metadata[i]["tokens"] = self.token_counter.count_tokens(
                chunks[i], 
                self.embedding_service.embedding_model
            )
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks from {file_path}")
        embeddings = self.embedding_service.get_embeddings(chunks)
        
        # Add to vector store
        self.vector_store.add_texts(chunks, embeddings, metadata)
        
        logger.info(f"Indexed {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def clear_index(self) -> None:
        """Clear the current index, texts, and metadata."""
        self.vector_store.clear()
        self.query_cache.clear()
    
    # ------------------ Save and Load Functions ------------------
    
    def save_index(self, directory: str = "vector_index") -> None:
        """
        Save the index, texts, and metadata to disk.
        
        Args:
            directory: Directory to save the index files.
        """
        self.vector_store.save(directory)
    
    def load_index(self, directory: str = "vector_index") -> bool:
        """
        Load the index, texts, and metadata from disk.
        
        Args:
            directory: Directory containing the index files.
            
        Returns:
            True if successful, False otherwise.
        """
        success = self.vector_store.load(directory)
        if success:
            # Clear query cache when loading a new index
            self.query_cache.clear()
        return success
    
    # ------------------ Search and Retrieval Functions ------------------
    
    def search(self, query: str, k: int = 3, use_hybrid: bool = True, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Search for most similar documents to a query with improved options.
        
        Args:
            query: The search query.
            k: Number of results to return.
            use_hybrid: Whether to use hybrid retrieval (vector + sparse).
            rerank: Whether to rerank results.
            
        Returns:
            List of dictionaries with search results.
        """
        # Check cache first if enabled - FIXED WITH HASH STRING
        cache_key = hashlib.sha256(f"{query}:{k}:{use_hybrid}:{rerank}".encode()).hexdigest()
        if self.cache_enabled and cache_key in self.query_cache:
            logger.info("Using cached search results")
            return self.query_cache[cache_key]
        
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            query_text=query,
            k=k,
            use_hybrid=use_hybrid,
            rerank=rerank
        )
        
        # Cache results if enabled
        if self.cache_enabled:
            self.query_cache[cache_key] = results
        
        return results
    
    def search_with_metadata_filter(self, query: str, filter_key: str, filter_value: Any, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents with a metadata filter.
        
        Args:
            query: The search query.
            filter_key: Metadata key to filter on.
            filter_value: Value to match for the filter key.
            k: Number of results to return.
            
        Returns:
            List of dictionaries with filtered search results.
        """
        # Check cache first if enabled - FIXED WITH HASH STRING
        cache_key = hashlib.sha256(f"{query}:{filter_key}:{str(filter_value)}:{k}".encode()).hexdigest()
        if self.cache_enabled and cache_key in self.query_cache:
            logger.info("Using cached filtered search results")
            return self.query_cache[cache_key]
        
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Search with filter
        results = self.vector_store.search_with_filter(
            query_embedding=query_embedding,
            query_text=query,
            filter_key=filter_key,
            filter_value=filter_value,
            k=k
        )
        
        # Cache results if enabled
        if self.cache_enabled:
            self.query_cache[cache_key] = results
        
        return results
    
    def get_document_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its chunk ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve.
            
        Returns:
            Dictionary with document info or None if not found.
        """
        return self.vector_store.get_document_by_id(chunk_id)
    
    def get_context_window(self, chunk_id: int, window_size: int = 1) -> List[Dict[str, Any]]:
        """
        Get surrounding context documents for a specific chunk.
        
        Args:
            chunk_id: ID of the center chunk.
            window_size: Number of documents to include on each side.
            
        Returns:
            List of context documents.
        """
        return self.vector_store.get_context_window(chunk_id, window_size)
    
    # ------------------ LLM Integration Functions ------------------
    
    def generate_answer(self, question: str, context_texts: List[str], 
                        model: str = "gpt-4.1-turbo", 
                        temperature: float = 0.0,
                        max_tokens: int = 1000) -> str:
        """
        Generate an answer using the LLM with improved prompt.
        
        Args:
            question: The question to answer.
            context_texts: List of context texts to use.
            model: The LLM model to use.
            temperature: Temperature for response generation.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            Generated answer as a string.
        """
        # Count tokens to make sure we don't exceed context limit
        prompt_tokens = 0
        formatted_contexts = []
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        - Answer accurately and directly based ONLY on the information in the context.
        - If the context doesn't contain the answer, say "I don't have enough information to answer this question" and suggest what information might help.
        - Cite your sources using [Source X] notation when drawing information from the provided context.
        - Never make up information that isn't supported by the context.
        - Format your answer clearly and concisely.
        - If the context contains code, preserve code formatting in your answer."""
        
        # Estimate tokens for system prompt and question
        prompt_tokens += self.token_counter.count_tokens(system_prompt, model)
        prompt_tokens += self.token_counter.count_tokens(question, model)
        
        # Add formatting tokens estimate
        prompt_tokens += 200  # Buffer for formatting
        
        # Add context pieces until we approach token limit
        max_context_tokens = 8000 - prompt_tokens - max_tokens
        current_context_tokens = 0
        
        # Score and sort context texts by relevance to question
        context_with_scores = []
        for i, context in enumerate(context_texts):
            # Simple relevance scoring: count overlapping terms
            question_terms = set(question.lower().split())
            context_terms = set(context.lower().split())
            overlap = len(question_terms.intersection(context_terms))
            
            # Exact phrase matching
            exact_match = 1 if question.lower() in context.lower() else 0
            
            # Calculate score (prioritize exact matches and term overlap)
            score = exact_match * 10 + overlap
            
            context_with_scores.append((context, score, i))
        
        # Sort contexts by score (descending)
        context_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add contexts until we hit the token limit
        for context, score, original_idx in context_with_scores:
            context_tokens = self.token_counter.count_tokens(context, model)
            
            if current_context_tokens + context_tokens <= max_context_tokens:
                formatted_contexts.append(f"[Source {original_idx+1}]\n{context}")
                current_context_tokens += context_tokens
            else:
                # If we're about to exceed the limit, break
                logger.info(f"Stopping at {len(formatted_contexts)} context chunks to avoid exceeding token limit")
                break
        
        logger.info(f"Using {len(formatted_contexts)}/{len(context_texts)} context chunks, ~{current_context_tokens} tokens")
        
        # Prepare the prompt
        full_context = "\n\n".join(formatted_contexts)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {question}"}
        ]
        
        # Generate answer
        return self.embedding_service.generate_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def query(self, question: str, k: int = 5, model: str = "gpt-4.1-turbo") -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to answer.
            k: Number of documents to retrieve.
            model: The LLM model to use for answer generation.
            
        Returns:
            Dictionary with answer and source documents.
        """
        # Check if index is loaded
        if not self.vector_store.texts:
            return {"answer": "No indexed documents. Please index some documents first."}
        
        # Check cache first - FIXED WITH HASH STRING
        cache_key = hashlib.sha256(
            json.dumps({
                "question": question,
                "k": k,
                "model": model
            }, sort_keys=True).encode()
        ).hexdigest()
        
        if self.cache_enabled and cache_key in self.query_cache:
            logger.info("Using cached query results")
            return self.query_cache[cache_key]
        
        # Search for relevant documents
        search_results = self.search(question, k=k*2, use_hybrid=True, rerank=True)
        
        if not search_results:
            return {"answer": "No relevant documents found."}
        
        # Get context window for each result to provide more context
        expanded_texts = []
        seen_ids = set()
        
        for result in search_results[:k]:  # Use top k results only
            chunk_id = result["metadata"].get("chunk_id")
            if chunk_id is not None:
                # Get the document and its context window
                context_docs = self.get_context_window(chunk_id, window_size=1)
                
                # Add unique documents
                for doc in context_docs:
                    doc_id = doc["metadata"].get("chunk_id")
                    if doc_id not in seen_ids:
                        expanded_texts.append(doc["text"])
                        seen_ids.add(doc_id)
        
        # If we couldn't get context windows, fall back to search results
        if not expanded_texts:
            expanded_texts = [result["text"] for result in search_results[:k]]
        
        # Generate answer
        answer = self.generate_answer(question, expanded_texts, model)
        
        # Record query in history for evaluation
        query_record = {
            "question": question,
            "timestamp": time.time(),
            "search_results": search_results[:k],
            "expanded_context": len(expanded_texts),
            "answer": answer
        }
        
        result = {
            "answer": answer,
            "source_documents": search_results[:k],
            "query_record": query_record
        }
        
        # Cache the result if enabled
        if self.cache_enabled:
            self.query_cache[cache_key] = result
        
        return result
    
    # ------------------ Evaluation Functions ------------------
    
    def evaluate_retrieval(self, questions: List[str], expected_chunks: List[List[int]], k: int = 5) -> Dict[str, Any]:
        """
        Evaluate retrieval performance using recall@k.
        
        Args:
            questions: List of test questions.
            expected_chunks: List of lists of expected chunk IDs for each question.
            k: Number of results to retrieve.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        return Evaluation_Utils.evaluate_retrieval(
            retriever=self.vector_store,
            embedding_function=self.embedding_service.get_embedding,
            questions=questions,
            expected_chunks=expected_chunks,
            k=k
        )
    
    # ------------------ Utility Functions ------------------
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the loaded index.
        
        Returns:
            Dictionary with index statistics.
        """
        # Get vector store stats
        store_stats = self.vector_store.get_stats()
        
        # Add embedding model info
        return {
            **store_stats,
            "embedding_model": self.embedding_service.embedding_model,
            "cache_enabled": self.cache_enabled,
            "query_cache_size": len(self.query_cache) if self.cache_enabled else 0
        }
    
    def is_loaded(self) -> bool:
        """
        Check if index is loaded.
        
        Returns:
            True if the index is loaded, False otherwise.
        """
        return self.vector_store.index is not None and bool(self.vector_store.texts)
        
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self.cache_enabled:
            self.query_cache.clear()
            logger.info("Query cache cleared")
            
            # Clear embedding service cache too
            if hasattr(self.embedding_service, 'embedding_cache'):
                self.embedding_service.embedding_cache.clear()
                logger.info("Embedding cache cleared")
                
    def optimize_index(self) -> None:
        """Perform optimization on the vector index (if supported)."""
        # This method could implement index optimization strategies
        # like clustering, quantization, or periodic reindexing
        # Currently just logs a message
        logger.info("Index optimization is not implemented for the current vector store backend")


    # Add these methods to your RAGController class

    def clear(self):
        """Clear all documents from the vector store"""
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            # Implementation depends on your vector store type
            if hasattr(self.vector_store, 'clear'):
                self.vector_store.clear()
            else:
                # Alternative approach if no clear method exists
                self.vector_store = None
                self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize or reinitialize the vector store"""
        from Cloud_LLM_Model.RAG.Retrieval.vector_store import VectorStore
        
        # Create a new empty vector store (adjust based on your implementation)
        self.vector_store = VectorStore(
            embedding_service=self.embedding_service,
            cache_enabled=self.cache_enabled
        )
        
        # Save the empty index
        self.save_index('vector_index')