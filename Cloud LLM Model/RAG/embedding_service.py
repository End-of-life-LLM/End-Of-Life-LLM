"""Service for generating and managing text embeddings."""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from Utils.token_counter import Token_Counter
from Utils.rate_limiter import Rate_Limiter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Embedding_Service:
    """Service for generating and managing text embeddings."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-large",
                 tier: str = "free"):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            embedding_model: Name of the embedding model to use.
            tier: API tier for rate limiting ('free' or 'tier1').
        """
        self.api_key = api_key
        if not self.api_key:
            from dotenv import load_dotenv
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it directly or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = embedding_model
        
        # Set up token counter
        self.token_counter = Token_Counter()
        
        # Set up rate limiter based on tier
        if tier == "free":
            self.rate_limiter = Rate_Limiter(
                requests_per_minute=100,
                tokens_per_minute=40000,
                max_batch_size=8
            )
        else:  # tier1
            self.rate_limiter = Rate_Limiter(
                requests_per_minute=3000,
                tokens_per_minute=1000000,
                max_batch_size=16
            )
        
        # Embedding dimension based on model
        if "3-large" in embedding_model:
            self.dimension = 3072
        elif "3-small" in embedding_model:
            self.dimension = 1536
        elif "ada-002" in embedding_model:
            self.dimension = 1536
        else:
            self.dimension = 1536  # Default fallback
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text with retry logic.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        # Empty text check
        if not text or text.strip() == "":
            # Return a zero vector of appropriate dimension
            return [0.0] * self.dimension
            
        # Count tokens for rate limiting
        token_count = self.token_counter.count_tokens(text, self.embedding_model)
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(token_count)
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            NumPy array of embeddings.
        """
        if not texts:
            return np.array([])
        
        # Determine optimal batch size
        batch_size = self.rate_limiter.get_optimal_batch_size(
            texts, 
            self.token_counter,
            self.embedding_model
        )
        logger.info(f"Using batch size of {batch_size} for {len(texts)} texts")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Empty text check for batch
            embeddings = np.zeros((len(batch), self.dimension), dtype=np.float32)
            
            # Filter out empty texts
            valid_texts = [text for text in batch if text and text.strip() != ""]
            if valid_texts:
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=valid_texts
                    )
                    
                    # Fill in valid embeddings
                    valid_idx = 0
                    for j, text in enumerate(batch):
                        if text and text.strip() != "":
                            embeddings[j] = response.data[valid_idx].embedding
                            valid_idx += 1
                except Exception as e:
                    logger.error(f"Error getting batch embeddings: {e}")
                    # Fall back to individual embeddings
                    for j, text in enumerate(batch):
                        if text and text.strip() != "":
                            embeddings[j] = self.get_embedding(text)
            
            all_embeddings.append(embeddings)
            
            # Progress logging
            logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return np.vstack(all_embeddings).astype('float32')
    
    def generate_chat_completion(self, 
                                messages: List[Dict[str, str]], 
                                model: str = "gpt-4.1-turbo", 
                                temperature: float = 0.0,
                                max_tokens: int = 1000) -> str:
        """
        Generate a response using the chat completion API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: The model to use.
            temperature: Temperature for response generation.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            Generated response as a string.
        """
        # Count tokens for rate limiting
        token_count = self.token_counter.count_messages_tokens(messages, model)
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(token_count)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            return f"Error generating response: {str(e)}"