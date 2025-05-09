"""Service for generating and managing text embeddings."""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from Cloud_LLM_Model.Utils.token_counter import Token_Counter
from Cloud_LLM_Model.Utils.rate_limiter import Rate_Limiter
from functools import lru_cache
import hashlib
import pickle
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global embedding cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class Embedding_Service:
    """Service for generating and managing text embeddings."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-large",
                 tier: str = "free",
                 cache_enabled: bool = True,
                 cache_ttl: int = 7*24*60*60):  # 1 week default TTL
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            embedding_model: Name of the embedding model to use.
            tier: API tier for rate limiting ('free' or 'tier1').
            cache_enabled: Whether to enable embedding caching.
            cache_ttl: Time-to-live for cache entries in seconds.
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
            
        # Cache settings
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache_file = os.path.join(CACHE_DIR, f"embedding_cache_{embedding_model.replace('-', '_')}.pkl")
        
        # Initialize cache
        self.embedding_cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Tuple[List[float], float]]:
        """
        Load embedding cache from disk.
        
        Returns:
            Dictionary mapping text hashes to (embedding, timestamp) tuples.
        """
        if not self.cache_enabled:
            return {}
            
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings")
                
                # Clean expired entries
                current_time = time.time()
                clean_cache = {k: v for k, v in cache.items() 
                              if current_time - v[1] < self.cache_ttl}
                
                if len(clean_cache) < len(cache):
                    logger.info(f"Removed {len(cache) - len(clean_cache)} expired cache entries")
                    
                return clean_cache
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.cache_enabled or not self.embedding_cache:
            return
            
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Generate a hash for text to use as cache key.
        
        Args:
            text: Text to hash.
            
        Returns:
            SHA-256 hash of the text.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text with retry logic and caching.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        # Empty text check
        if not text or text.strip() == "":
            # Return a zero vector of appropriate dimension
            return [0.0] * self.dimension
        
        # Check cache first if enabled
        if self.cache_enabled:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                embedding, timestamp = self.embedding_cache[text_hash]
                
                # Check if entry is still valid
                if time.time() - timestamp < self.cache_ttl:
                    return embedding
                # Otherwise, it's expired and we'll generate a new embedding
        
        # Count tokens for rate limiting
        token_count = self.token_counter.count_tokens(text, self.embedding_model)
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(token_count)
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the embedding if enabled
            if self.cache_enabled:
                text_hash = self._get_text_hash(text)
                self.embedding_cache[text_hash] = (embedding, time.time())
                
                # Periodically save cache (every 100 new entries)
                if len(self.embedding_cache) % 100 == 0:
                    self._save_cache()
                
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts with batching and caching.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            NumPy array of embeddings.
        """
        if not texts:
            return np.array([])
        
        # Initialize array for results
        all_embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        # Check which texts we already have in cache
        if self.cache_enabled:
            cache_hits = 0
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if not text or text.strip() == "":
                    # For empty text, use zero vector
                    all_embeddings[i] = np.zeros(self.dimension, dtype=np.float32)
                    continue
                    
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache:
                    embedding, timestamp = self.embedding_cache[text_hash]
                    
                    # Check if entry is still valid
                    if time.time() - timestamp < self.cache_ttl:
                        all_embeddings[i] = embedding
                        cache_hits += 1
                        continue
                
                # If not in cache or expired, add to list for embedding
                texts_to_embed.append(text)
                text_indices.append(i)
                
            logger.info(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits*100/len(texts):.1f}%)")
        else:
            # If cache is disabled, embed all non-empty texts
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if not text or text.strip() == "":
                    # For empty text, use zero vector
                    all_embeddings[i] = np.zeros(self.dimension, dtype=np.float32)
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        
        # If we have texts to embed
        if texts_to_embed:
            # Determine optimal batch size
            batch_size = self.rate_limiter.get_optimal_batch_size(
                texts_to_embed, 
                self.token_counter,
                self.embedding_model
            )
            logger.info(f"Using batch size of {batch_size} for {len(texts_to_embed)} texts")
            
            # Process in batches
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i+batch_size]
                batch_indices = text_indices[i:i+batch_size]
                
                # Count tokens for rate limiting
                batch_tokens = sum(self.token_counter.count_tokens(text, self.embedding_model) for text in batch)
                
                # Apply rate limiting
                self.rate_limiter.wait_if_needed(batch_tokens)
                
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch
                    )
                    
                    # Process responses
                    for j, emb_data in enumerate(response.data):
                        idx = batch_indices[j]
                        all_embeddings[idx] = np.array(emb_data.embedding, dtype=np.float32)
                        
                        # Add to cache if enabled
                        if self.cache_enabled:
                            text_hash = self._get_text_hash(batch[j])
                            self.embedding_cache[text_hash] = (emb_data.embedding, time.time())
                except Exception as e:
                    logger.error(f"Error getting batch embeddings: {e}")
                    # Fall back to individual embeddings
                    for j, text in enumerate(batch):
                        idx = batch_indices[j]
                        all_embeddings[idx] = self.get_embedding(text)
                
                # Progress logging
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
            
            # Save cache after all batches are processed
            if self.cache_enabled:
                self._save_cache()
        
        return all_embeddings
    
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