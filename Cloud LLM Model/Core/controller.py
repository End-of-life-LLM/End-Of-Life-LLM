"""Main controller for the Cloud LLM Model system."""

from model import Model
from supporting_model import Supporting_Model
from RAG.controller import RAGController


class Controller:
    """
    Main controller class for the Cloud LLM Model system.
    Integrates the Model, Supporting_Model, and RAG subsystems.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Controller.
        
        Args:
            api_key: Optional OpenAI API key. If None, will try to load from environment.
        """
        self.model = Model(api_key=api_key)
        self.supporting_model = Supporting_Model(api_key=api_key)
        
        # Initialize RAG system
        self.rag_controller = RAGController(api_key=api_key)