"""Rate limiting utility for API calls."""

import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Rate_Limiter:
    """Class to manage API rate limits."""
    
    def __init__(self, 
                 requests_per_minute: int = 1000, 
                 tokens_per_minute: int = 40000,
                 max_batch_size: int = 16):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum API requests per minute.
            tokens_per_minute: Maximum tokens per minute.
            max_batch_size: Maximum batch size for embedding requests.
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_batch_size = max_batch_size
        
        self.request_timestamps = []
        self.token_usage = []
        
        # Time window in seconds
        self.time_window = 60
    
    def wait_if_needed(self, token_count: int = 0) -> None:
        """
        Wait if necessary to stay within rate limits.
        
        Args:
            token_count: Number of tokens in the current request.
        """
        current_time = time.time()
        
        # Clean up old timestamps outside the time window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts < self.time_window]
        self.token_usage = [(t, ts) for t, ts in self.token_usage 
                           if current_time - ts < self.time_window]
        
        # Check if we're at the request limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = self.time_window - (current_time - self.request_timestamps[0]) + 0.1
            logger.info(f"Rate limit approaching: Sleeping for {sleep_time:.2f}s")
            time.sleep(max(0, sleep_time))
        
        # Check if we're at the token limit
        current_token_usage = sum(tokens for tokens, _ in self.token_usage)
        if current_token_usage + token_count >= self.tokens_per_minute:
            sleep_time = self.time_window - (current_time - self.token_usage[0][1]) + 0.1
            logger.info(f"Token limit approaching: Sleeping for {sleep_time:.2f}s")
            time.sleep(max(0, sleep_time))
        
        # Record this request
        self.request_timestamps.append(time.time())
        if token_count > 0:
            self.token_usage.append((token_count, time.time()))
    
    def get_optimal_batch_size(self, texts: List[str], token_counter, model: str) -> int:
        """
        Calculate optimal batch size based on token count and rate limits.
        
        Args:
            texts: List of texts to process.
            token_counter: TokenCounter instance for counting tokens.
            model: Model to use for token counting.
            
        Returns:
            Optimal batch size.
        """
        if len(texts) <= 1:
            return 1
        
        # Sample a few texts to estimate average token count
        sample_size = min(5, len(texts))
        samples = [texts[i] for i in range(0, len(texts), max(1, len(texts) // sample_size))]
        
        avg_tokens = sum(token_counter.count_tokens(text, model) for text in samples) / len(samples)
        
        # Calculate how many texts we can process per minute based on token limit
        texts_per_minute = min(
            self.requests_per_minute,
            int(self.tokens_per_minute / avg_tokens)
        )
        
        # Determine batch size (don't exceed max_batch_size)
        optimal_batch_size = min(
            self.max_batch_size,
            max(1, int(texts_per_minute / (self.requests_per_minute / 3)))  # Use 1/3 of request limit for safety
        )
        
        return optimal_batch_size