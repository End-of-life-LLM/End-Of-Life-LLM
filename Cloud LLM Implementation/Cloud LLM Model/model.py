from openai import OpenAI


class Model:
    def __init__(self, api_key: str): 
        
        self.temperature = 0.7
        self.messages = []
        self.client = OpenAI(api_key=api_key)


    def context_manager(self):
        pass
    

    def message_manager(self) -> bool:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            return False
        
        self.messages.append({"role": "user", "content": user_input})

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