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
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model with retry logic for connection errors.
        
        Args:
            prompt: The prompt text to send to the model
            
        Returns:
            A string containing the model's response
        """
        import time  # Make sure this is imported
        
        # Add prompt to messages
        self.messages.append({"role": "user", "content": prompt})
        self.context_manager()
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Make API call with timeout
                completion = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=1000,  # Increased for more complete responses
                    timeout=30  # Add timeout in seconds
                )
            
                # Get assistant's response
                assistant_response = completion.choices[0].message.content
                
                # Add assistant's response to conversation history
                self.messages.append({"role": "assistant", "content": assistant_response})
                
                return assistant_response
                
            except Exception as e:
                error_msg = str(e)
                
                # Log the specific error
                import logging
                logging.error(f"API call attempt {retry_count+1} failed: {error_msg}")
                
                # Check if it's a connection error
                if "Connection error" in error_msg:
                    retry_count += 1
                    if retry_count < max_retries:
                        # Wait before retrying (exponential backoff)
                        backoff_time = 2 ** retry_count
                        logging.info(f"Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        continue
                else:
                    # If it's not a connection error, don't retry
                    return f"Error: {error_msg}"
                    
        return f"Failed to generate response after {max_retries} attempts due to connection issues."