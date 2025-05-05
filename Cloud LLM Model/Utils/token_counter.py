"""Token counting utility for various LLM models."""

from typing import Any, Dict, List
import tiktoken


class Token_Counter:
    """Utility class to count tokens for various models using tiktoken."""
    
    def __init__(self):
        """Initialize token counter with model-to-encoding mappings."""
        # We'll create encoders on demand to avoid loading all at startup
        self.encoders = {}
        
        # Maps models to their encoding names
        self.model_to_encoding = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-4.1-turbo": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-3-large": "cl100k_base"
        }
    
    def _get_encoder(self, model: str):
        """
        Get the appropriate encoder for a model.
        
        Args:
            model: Model name to get encoder for.
            
        Returns:
            Tiktoken encoder for the specified model.
        """
        # Check if we already have this encoder
        if model in self.encoders:
            return self.encoders[model]
        
        # Get the encoding name for this model
        encoding_name = self.model_to_encoding.get(model)
        
        # If we don't have a specific mapping, try getting it directly or use a default
        if not encoding_name:
            try:
                # Try to get the encoding directly from the model name
                encoder = tiktoken.encoding_for_model(model)
                self.encoders[model] = encoder
                return encoder
            except KeyError:
                # Fall back to cl100k_base for newer models
                encoding_name = "cl100k_base"
        
        # Create and cache the encoder
        encoder = tiktoken.get_encoding(encoding_name)
        self.encoders[model] = encoder
        return encoder
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count the number of tokens in a text for a specific model.
        
        Args:
            text: The text to count tokens for.
            model: The model to use for counting tokens.
            
        Returns:
            Number of tokens.
        """
        if not text:
            return 0
            
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
        """
        Count tokens in a list of chat messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: The model to use for counting tokens.
            
        Returns:
            Total token count including message formatting.
        """
        # Base tokens for message formatting
        num_tokens = 0
        
        # Token counts for each message
        for message in messages:
            # Add tokens for message role and content
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            
            for key, value in message.items():
                if not value:
                    continue
                    
                num_tokens += self.count_tokens(str(value), model)
                if key == "name":  # If there's a name, the role is omitted
                    num_tokens -= 1  # Role is omitted
                    
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens