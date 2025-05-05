"""Main controller for Retrieval Augmented Generation"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Import from our modules
from embedding_service import Embedding_Service
from Utils.token_counter import Token_Counter
from Retrieval.vector_store import Vector_Store
from Text_Utils.evaluation_utils import Evaluation_Utils
from Text_Utils.text_processor import Text_Processor

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
                tier: str = "free"):
        """
        Initialize the RAG controller.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            embedding_model: Name of the embedding model to use.
            tier: API tier for rate limiting ('free' or 'tier1').
        """
        # Initialize embedding service
        self.embedding_service = Embedding_Service(
            api_key=api_key,
            embedding_model=embedding_model,
            tier=tier
        )
        
        # Initialize vector store with the same dimension as embedding model
        self.vector_store = Vector_Store(dimension=self.embedding_service.dimension)
        
        # Keep a token counter instance for convenience
        self.token_counter = Token_Counter()
        
        # Create text processor instance
        self.text_processor = Text_Processor()
    
    # ------------------ Indexing Functions ------------------
    
    def index_text_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """
        Index a text file by splitting, embedding, and storing in vector store.
        
        Args:
            file_path: Path to the text file.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            Number of chunks indexed.
        """
        # Chunk the text file
        chunks, metadata = self.text_processor.read_and_chunk_file(file_path, chunk_size, chunk_overlap)
        
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
        return self.vector_store.load(directory)
    
    # ------------------ Search and Retrieval Functions ------------------
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for most similar documents to a query.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            List of dictionaries with search results.
        """
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Search in vector store
        return self.vector_store.search(
            query_embedding=query_embedding,
            query_text=query,
            k=k
        )
    
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
        # Get query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Search with filter
        return self.vector_store.search_with_filter(
            query_embedding=query_embedding,
            query_text=query,
            filter_key=filter_key,
            filter_value=filter_value,
            k=k
        )
    
    def get_document_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its chunk ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve.
            
        Returns:
            Dictionary with document info or None if not found.
        """
        return self.vector_store.get_document_by_id(chunk_id)
    
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
        - Format your answer clearly and concisely."""
        
        # Estimate tokens for system prompt and question
        prompt_tokens += self.token_counter.count_tokens(system_prompt, model)
        prompt_tokens += self.token_counter.count_tokens(question, model)
        
        # Add formatting tokens estimate
        prompt_tokens += 200  # Buffer for formatting
        
        # Add context pieces until we approach token limit
        max_context_tokens = 8000 - prompt_tokens - max_tokens
        current_context_tokens = 0
        
        for i, context in enumerate(context_texts):
            context_tokens = self.token_counter.count_tokens(context, model)
            
            if current_context_tokens + context_tokens <= max_context_tokens:
                formatted_contexts.append(f"[Source {i+1}]\n{context}")
                current_context_tokens += context_tokens
            else:
                # If we're about to exceed the limit, break
                logger.info(f"Stopping at {i} context chunks to avoid exceeding token limit")
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
        
        # Search for relevant documents
        search_results = self.search(question, k=k)
        
        if not search_results:
            return {"answer": "No relevant documents found."}
        
        # Extract texts from search results
        context_texts = [result["text"] for result in search_results]
        
        # Generate answer
        answer = self.generate_answer(question, context_texts, model)
        
        # Record query in history for evaluation
        query_record = {
            "question": question,
            "timestamp": time.time(),
            "search_results": search_results,
            "answer": answer
        }
        
        return {
            "answer": answer,
            "source_documents": search_results,
            "query_record": query_record
        }
    
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
            "embedding_model": self.embedding_service.embedding_model
        }
    
    def is_loaded(self) -> bool:
        """
        Check if index is loaded.
        
        Returns:
            True if the index is loaded, False otherwise.
        """
        return self.vector_store.index is not None and bool(self.vector_store.texts)