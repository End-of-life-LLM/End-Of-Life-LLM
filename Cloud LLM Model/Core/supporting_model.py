"""Supporting model interface for the Cloud LLM Model system."""

from openai import OpenAI


class Supporting_Model:
    """Supporting model for auxiliary tasks."""
    
    def __init__(self, api_key=None):
        """
        Initialize the supporting model.
        
        Args:
            api_key: Optional API key.
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        
    def generate_embeddings(self, text):
        """
        Generate embeddings for a text.
        This is a simple example and would be replaced with actual implementations.
        
        Args:
            text: Text to generate embeddings for.
            
        Returns:
            Embeddings for the input text.
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None