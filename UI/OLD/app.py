from flask import Flask, request, jsonify
from flask_cors import CORS  # You'll need to install this: pip install flask-cors

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Function 1: Process incoming message
def process_message(message):
    """
    This function accepts a string (the user's message)
    In a real implementation, you would add your processing logic here
    """
    return message

# Function 2: Generate response
def generate_response(message):
    """
    This function returns a string to show as the bot's response
    In a real implementation, you would connect to an AI service or other logic
    """
    return f"You said: {message}. This is a simple echo response."

# Flask route to handle messages
@app.route('/send_message', methods=['POST'])
def handle_message():
    data = request.json
    user_message = data.get('message', '')
    
    # Use Function 1 to process the message
    processed_message = process_message(user_message)
    
    # Use Function 2 to generate a response
    response = generate_response(processed_message)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)