"""Primary model interface for the Cloud LLM Model system."""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Model:
    """Primary model interface using OpenAI API."""
    
    def __init__(self, api_key=None, model_name="gpt-4.1", temperature=0.7):
        """
        Initialize the model.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            model_name: Model name to use. Defaults to "gpt-4.1".
            temperature: Temperature for generation. Defaults to 0.7.
        """
        # Use provided API key or fall back to environment variable
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide one or set OPENAI_API_KEY in env.")
        
        self.temperature = temperature
        self.model_name = model_name
        self.messages = []
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize logger
        self.logger = logging.getLogger("Model")
        self.logger.info(f"Model initialized with model_name={model_name}")

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
                model=self.model_name,
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
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, timeout: int = 30) -> str:
        """
        Generate a response from the model with retry logic for connection errors.
        Includes instructions for plain text output.
        
        Args:
            prompt: The prompt text to send to the model
            max_tokens: Maximum tokens in the response. Defaults to 1000.
            timeout: Timeout in seconds. Defaults to 30.
            
        Returns:
            A string containing the model's response
        """
        # Add system message to enforce plain text formatting
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that provides information in plain text format only. Do not use any fancy formatting such as bold text, tables, markdown formatting, or bullet points. Use simple paragraphs with clear language."
        }
        
        # Add prompt to messages
        user_message = {"role": "user", "content": prompt}
        
        # Create messages array with system message
        messages = [system_message, user_message]
        
        # Store in conversation history
        self.context_manager()
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Make API call with timeout
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
            
                # Get assistant's response
                assistant_response = completion.choices[0].message.content
                
                # Add assistant's response to conversation history
                self.messages.append({"role": "assistant", "content": assistant_response})
                
                return assistant_response
                
            except Exception as e:
                error_msg = str(e)
                
                # Log the specific error
                self.logger.error(f"API call attempt {retry_count+1} failed: {error_msg}")
                
                # Check if it's a connection error
                if "Connection error" in error_msg:
                    retry_count += 1
                    if retry_count < max_retries:
                        # Wait before retrying (exponential backoff)
                        backoff_time = 2 ** retry_count
                        self.logger.info(f"Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        continue
                else:
                    # If it's not a connection error, don't retry
                    return f"Error: {error_msg}"
                    
        return f"Failed to generate response after {max_retries} attempts due to connection issues."
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.messages = []
        self.logger.info("Conversation history reset")
    
    def set_temperature(self, temperature: float):
        """
        Set the temperature for generation.
        
        Args:
            temperature: Temperature value between 0 and 1.
        """
        if 0 <= temperature <= 1:
            self.temperature = temperature
            self.logger.info(f"Temperature set to {temperature}")
        else:
            self.logger.warning(f"Invalid temperature value {temperature}. Value should be between 0 and 1.")
    
    def set_model(self, model_name: str):
        """
        Set the model to use.
        
        Args:
            model_name: Name of the model to use.
        """
        self.model_name = model_name
        self.logger.info(f"Model set to {model_name}")