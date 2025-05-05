"""Primary model interface for the Cloud LLM Model system."""

from openai import OpenAI


class Model:
    """Primary model interface using OpenAI API."""
    
    def __init__(self, api_key=None): 
        """
        Initialize the model.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
        """
        self.temperature = 0.7
        self.messages = []
        self.client = OpenAI(api_key=api_key)

    def context_manager(self):
        """Manage context window for the model."""
        # Implementation for managing context window size
        if len(self.messages) > 10:
            # Keep only the last 10 messages
            self.messages = self.messages[-10:]
    
    def message_manager(self) -> bool:
        """
        Manage messages and interact with the model.
        
        Returns:
            True if the conversation should continue, False otherwise.
        """
        user_input = input("You: ")
        if user_input.lower() == "exit":
            return False
        
        self.messages.append({"role": "user", "content": user_input})
        self.context_manager()

        try:
            # Make API call
            completion = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=200
            )
        
            # Get assistant's response
            assistant_response = completion.choices[0].message.content
        
            # Print assistant's response
            print(f"Assistant: {assistant_response}")
        
            # Add assistant's response to conversation history
            self.messages.append({"role": "assistant", "content": assistant_response})
        
        except Exception as e:
            print(f"Error: {e}")
        return True