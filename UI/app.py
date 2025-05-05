from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")





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