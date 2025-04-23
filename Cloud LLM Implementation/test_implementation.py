from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize conversation history
messages = []

print("Chat with GPT-4o-mini (type 'exit' to quit)")
print("-" * 40)

while True:
    # Get user input
    user_input = input("You: ")
    
    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})
    
    try:
        # Make API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        # Get assistant's response
        assistant_response = completion.choices[0].message.content
        
        # Print assistant's response
        print(f"Assistant: {assistant_response}")
        
        # Add assistant's response to conversation history
        messages.append({"role": "assistant", "content": assistant_response})
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("-" * 40)