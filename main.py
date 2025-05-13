"""
Main application file for the Cloud LLM Model system.
Runs a Flask web server to provide a chat interface.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, session, redirect
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

# Create .env file if it doesn't exist
env_file_path = os.path.join(os.getcwd(), '.env')
if not os.path.exists(env_file_path):
    logger.info("Creating .env file")
    with open(env_file_path, 'w') as f:
        f.write("# Configuration for Cloud LLM Model system\n")
        f.write("# Add your OpenAI API key below\n")
        f.write("# OPENAI_API_KEY=your-api-key-here\n")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Log API key status (without revealing the key)
if api_key:
    logger.info("API key found in environment variables")
else:
    logger.warning("No API key found in environment variables")

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

# Import our main controller
from Cloud_LLM_Model.Core.controller import Controller

# Initialize main controller with API key from environment
controller = Controller(api_key=api_key)

# Add a check_api_key middleware function to ensure API key is present for certain routes
def check_api_key():
    # Routes that don't require an API key check
    exempt_routes = ['/setup', '/save_api_settings', '/static/', '/favicon.ico']
    
    # Check if the current request path is exempt
    for route in exempt_routes:
        if request.path.startswith(route):
            return
    
    # Check if API key exists
    global api_key, controller
    if not api_key:
        session_data = controller.get_or_create_session(session.get('session_id'))
        if not session_data or 'settings' not in session_data or 'apiKey' not in session_data['settings'] or not session_data['settings']['apiKey']:
            # No API key found, redirect to setup
            return redirect('/setup')
        else:
            # Use API key from session to initialize components if needed
            session_api_key = session_data['settings']['apiKey']
            if not controller.model or not controller.rag_controller:
                success = controller.initialize_components(session_api_key)
                if not success:
                    # Initialization failed, redirect to setup
                    return redirect('/setup')

# Register the middleware function with Flask
@app.before_request
def before_request():
    return check_api_key()

@app.route('/')
def index():
    """Render the main chat interface or redirect to setup if API key is missing"""
    # Check if API key exists
    global api_key, controller
    
    # If no environment variable, check if there's an active session with an API key
    if not api_key:
        session_data = controller.get_or_create_session(session.get('session_id'))
        if session_data and 'settings' in session_data and 'apiKey' in session_data['settings'] and session_data['settings']['apiKey']:
            # Session has an API key, so we can proceed
            # Initialize model and RAG system if needed
            if not controller.model or not controller.rag_controller:
                success = controller.initialize_components(session_data['settings']['apiKey'])
                if not success:
                    # Initialization failed, redirect to setup
                    return redirect('/setup')
            return render_template('index.html')
        else:
            # No API key found, redirect to setup
            return redirect('/setup')
    
    # API key exists in environment, proceed normally
    return render_template('index.html')

@app.route('/home')
def home():
    """Render the main chat interface (alias) or redirect to setup if API key is missing"""
    return index()

@app.route('/setup')
def setup():
    """Render the setup page"""
    return render_template('setup.html')

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html')

@app.route('/save_api_settings', methods=['POST'])
def save_api_settings():
    """Save API settings and handle file uploads"""
    global controller
    
    # Get current session
    session_data = controller.get_or_create_session(session.get('session_id'))
    session['session_id'] = session_data['id']
    
    # Prepare form data
    form_data = {
        'session_id': session_data['id'],
        'apiKey': request.form.get('apiKey', ''),
        'temperature': request.form.get('temperature'),
        'maxTokens': request.form.get('maxTokens'),
        'timeout': request.form.get('timeout'),
        'numberOfFiles': request.form.get('numberOfFiles'),
        'urls': request.form.get('urls', '')
    }
    
    # Save settings via controller
    result = controller.save_api_settings(form_data, request.files.getlist('files[]'))
    
    # Create a response
    response = jsonify(result)
    
    # Set a cookie to store the API key
    if 'apiKey' in form_data and form_data['apiKey']:
        response.set_cookie('api_key', form_data['apiKey'], max_age=30*24*60*60)  # 30 days
    
    return response

@app.route('/send_message', methods=['POST'])
def send_message():
    """Process a user message and return AI response"""
    global controller
    
    data = request.json
    user_message = data.get('message', '')
    
    # Process message via controller
    result = controller.process_message(user_message, session.get('session_id'))
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    return jsonify(result)
    
@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Create a new chat session"""
    global controller
    
    # Create new session via controller
    session_data = controller.get_or_create_session()
    session['session_id'] = session_data['id']
    
    return jsonify({
        "success": True,
        "session_id": session_data['id']
    })

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Get the current chat history"""
    global controller
    
    session_data = controller.get_or_create_session(session.get('session_id'))
    session['session_id'] = session_data['id']
    
    return jsonify({
        "success": True,
        "messages": session_data['messages']
    })

@app.route('/get_chats', methods=['GET'])
def get_chats():
    """Get list of all chat sessions"""
    global controller
    
    # Use the file_manager's chat_sessions instead
    chats = []
    for chat_id, chat_data in controller.file_manager.chat_sessions.items():
        # Extract a preview from the first few messages or use title if available
        preview = chat_data.get('title', '')
        
        if not preview:
            for msg in chat_data['messages']:
                if msg['role'] == 'user':
                    preview = msg['content']
                    break
        
        if not preview and chat_data['messages']:
            preview = "New Chat"
        
        chats.append({
            "id": chat_id,
            "preview": preview[:50] + "..." if len(preview) > 50 else preview,
            "created_at": chat_data['created_at']
        })
    
    # Sort by creation time (newest first)
    chats.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({
        "success": True,
        "chats": chats
    })

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a specific chat session"""
    global controller
    
    result = controller.delete_chat(chat_id)
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    # If current session was deleted, clear the session cookie
    if session.get('session_id') == chat_id:
        session.pop('session_id', None)
    
    return jsonify(result)

@app.route('/clear_all_chats', methods=['DELETE'])
def clear_all_chats():
    """Delete all chat sessions"""
    global controller
    
    result = controller.clear_all_chats()
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    # Clear the session cookie
    session.pop('session_id', None)
    
    return jsonify(result)

@app.route('/rename_chat/<chat_id>', methods=['POST'])
def rename_chat(chat_id):
    """Rename a specific chat session"""
    global controller
    
    data = request.json
    new_title = data.get('title', '')
    
    result = controller.rename_chat(chat_id, new_title)
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    return jsonify(result)

@app.route('/system_info', methods=['GET'])
def system_info():
    """Get information about the system"""
    global controller
    
    # Get system info and add RAG toggle state
    system_info = controller.get_system_info()
    system_info["rag_enabled"] = controller.get_rag_enabled()["state"]
    
    return jsonify(system_info)

@app.route('/toggle', methods=['POST'])
def toggle():
    """Toggle the RAG system on/off"""
    global controller
    # Get current state and toggle it
    current_state = controller.get_rag_enabled()
    new_state = not current_state["state"]
    result = controller.set_rag_enabled(new_state)
    return jsonify(result)

@app.route('/get_state', methods=['GET'])
def get_state():
    """Get the current state of the RAG system"""
    global controller
    return jsonify(controller.get_rag_enabled())

@app.route('/get_indexed_files', methods=['GET'])
def get_indexed_files():
    """Get a list of all indexed files"""
    global controller
    
    result = controller.get_indexed_files(session.get('session_id'))
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    return jsonify(result)

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """Delete a specific indexed file"""
    global controller
    
    data = request.json
    filename = data.get('filename')
    
    result = controller.delete_file(filename, session.get('session_id'))
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    return jsonify(result)

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    """Delete all indexed files"""
    global controller
    
    result = controller.delete_all_files(session.get('session_id'))
    
    # Check if result is a tuple (error case)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    
    return jsonify(result)

if __name__ == '__main__':
    logger.info("Starting Cloud LLM Model system")
    
    # Final API key check
    if api_key:
        logger.info("API key is set. System should work correctly.")
    else:
        # Check if API key is in .env but wasn't loaded
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
                if 'OPENAI_API_KEY=' in env_content and not env_content.strip().endswith('OPENAI_API_KEY='):
                    logger.warning("API key found in .env file but wasn't loaded. Try restarting the application.")
                else:
                    logger.warning("No API key found. You will be prompted to enter one when accessing the application.")
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            logger.warning("No API key found. You will be prompted to enter one when accessing the application.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)