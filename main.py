"""
Main application file for the Cloud LLM Model system.
Runs a Flask web server to provide a chat interface.
"""

import os
import json
import time
import logging
import shutil
from typing import Dict, List, Any
from flask import Flask, render_template, request, jsonify, session, redirect, make_response
from dotenv import load_dotenv
import uuid

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

# Default settings
DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "model": "gpt-4.1"
}

# Import our system components
# Do this after environment variables are loaded
from Cloud_LLM_Model.Core.model import Model
from Cloud_LLM_Model.RAG.controller import RAGController
from WebSearshing.webArticleManger import WebArticleManager
from Cloud_LLM_Model.Utils.token_counter import Token_Counter

# Initialize global components
model = None
rag_system = None
try:
    if api_key:
        model = Model(api_key=api_key)
        rag_system = RAGController(api_key=api_key, embedding_model="text-embedding-3-large", cache_enabled=True)
        logger.info("Model and RAG system initialized with API key from environment")
    else:
        logger.warning("No API key found in environment. Model and RAG system will be initialized later.")
except Exception as e:
    logger.error(f"Error initializing model or RAG system: {e}")

article_manager = WebArticleManager(max_results=5, save_directory="articles")
token_counter = Token_Counter()
toggle_state = False

# Session management
CHAT_SESSIONS = {}  # Store chat sessions in memory (replace with database for production)

def get_or_create_session(session_id=None):
    """Get an existing session or create a new one"""
    if session_id and session_id in CHAT_SESSIONS:
        return CHAT_SESSIONS[session_id]
    
    # Create new session
    new_id = session_id or str(uuid.uuid4())
    CHAT_SESSIONS[new_id] = {
        "id": new_id,
        "created_at": time.time(),
        "messages": [],
        "settings": DEFAULT_SETTINGS.copy(),
        "indexed_files": [],
    }
    return CHAT_SESSIONS[new_id]

# Add a check_api_key middleware function to ensure API key is present for certain routes
def check_api_key():
    # Routes that don't require an API key check
    exempt_routes = ['/setup', '/save_api_settings', '/static/', '/favicon.ico']
    
    # Check if the current request path is exempt
    for route in exempt_routes:
        if request.path.startswith(route):
            return
    
    # Check if API key exists
    global api_key, model, rag_system
    if not api_key:
        session_data = get_or_create_session(session.get('session_id'))
        if not session_data or 'settings' not in session_data or 'apiKey' not in session_data['settings'] or not session_data['settings']['apiKey']:
            # No API key found, redirect to setup
            return redirect('/setup')
        else:
            # Use API key from session
            session_api_key = session_data['settings']['apiKey']
            
            # Initialize model and RAG system if needed
            if model is None or rag_system is None:
                try:
                    model = Model(api_key=session_api_key)
                    rag_system = RAGController(api_key=session_api_key, embedding_model="text-embedding-3-large", cache_enabled=True)
                    logger.info("Model and RAG system initialized with API key from session")
                except Exception as e:
                    logger.error(f"Error initializing model or RAG system: {e}")
                    # Redirect to setup if initialization fails
                    return redirect('/setup')

# Register the middleware function with Flask
@app.before_request
def before_request():
    return check_api_key()

@app.route('/')
def index():
    """Render the main chat interface or redirect to setup if API key is missing"""
    # Check if API key exists
    global api_key, model, rag_system
    
    # If no environment variable, check if there's an active session with an API key
    if not api_key:
        session_data = get_or_create_session(session.get('session_id'))
        if session_data and 'settings' in session_data and 'apiKey' in session_data['settings'] and session_data['settings']['apiKey']:
            # Session has an API key, so we can proceed
            # Initialize model and RAG system if needed
            if model is None or rag_system is None:
                try:
                    model = Model(api_key=session_data['settings']['apiKey'])
                    rag_system = RAGController(api_key=session_data['settings']['apiKey'], embedding_model="text-embedding-3-large", cache_enabled=True)
                    logger.info("Model and RAG system initialized with API key from session")
                except Exception as e:
                    logger.error(f"Error initializing model or RAG system: {e}")
                    # Redirect to setup if initialization fails
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
    global model, rag_system, api_key
    
    # Get current session
    session_data = get_or_create_session(session.get('session_id'))
    session['session_id'] = session_data['id']
    
    # Update settings
    if 'settings' not in session_data:
        session_data['settings'] = DEFAULT_SETTINGS.copy()
    
    settings = session_data['settings']
    new_api_key = request.form.get('apiKey', settings.get('apiKey', ''))
    settings['apiKey'] = new_api_key
    settings['temperature'] = float(request.form.get('temperature', settings.get('temperature', 0.7)))
    settings['max_tokens'] = int(request.form.get('maxTokens', settings.get('max_tokens', 1000)))
    settings['timeout'] = int(request.form.get('timeout', settings.get('timeout', 30)))
    settings['numberOfFiles'] = int(request.form.get('numberOfFiles', settings.get('numberOfFiles', 1)))

    # Handle file uploads
    files = request.files.getlist('files[]')
    indexed_files = []
    
    for file in files:
        if file and file.filename:
            # Save uploaded file to tmp directory
            tmp_dir = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            file_path = os.path.join(tmp_dir, file.filename)
            file.save(file_path)
    
    # If a new API key is provided, update the model and RAG system
    if new_api_key:
        try:
            # Update global API key for persistence
            api_key = new_api_key
            
            # Save API key to .env file for persistence across restarts
            env_file_path = os.path.join(os.getcwd(), '.env')
            
            # Check if .env file exists
            if os.path.exists(env_file_path):
                # Read existing .env file
                with open(env_file_path, 'r') as f:
                    env_lines = f.readlines()
                
                # Check if OPENAI_API_KEY already exists in the file
                key_exists = False
                for i, line in enumerate(env_lines):
                    if line.startswith('OPENAI_API_KEY='):
                        # Update existing key
                        env_lines[i] = f'OPENAI_API_KEY={new_api_key}\n'
                        key_exists = True
                        break
                
                # Append key if it doesn't exist
                if not key_exists:
                    env_lines.append(f'OPENAI_API_KEY={new_api_key}\n')
                
                # Write updated content back to .env file
                with open(env_file_path, 'w') as f:
                    f.writelines(env_lines)
            else:
                # Create new .env file
                with open(env_file_path, 'w') as f:
                    f.write(f'OPENAI_API_KEY={new_api_key}\n')
            
            logger.info("API key saved to .env file")
            
            # Initialize or reinitialize the model and RAG system
            model = Model(api_key=new_api_key, temperature=settings['temperature'])
            rag_system = RAGController(api_key=new_api_key, embedding_model="text-embedding-3-large", cache_enabled=True)
            logger.info("Model and RAG system initialized with new API key")
            
            # Try to load existing vector index if available
            try:
                if os.path.exists('vector_index'):
                    logger.info("Loading existing vector index...")
                    rag_system.load_index('vector_index')
                    logger.info(f"Vector index loaded: {rag_system.get_index_stats()}")
            except Exception as e:
                logger.error(f"Error loading vector index: {e}")
        except Exception as e:
            logger.error(f"Error initializing model or RAG system with new API key: {e}")
            return jsonify({
                "success": False,
                "message": f"Error initializing with the provided API key: {str(e)}"
            }), 400
    
    # Create a persistent cookie to store the API key
    response = jsonify({
        "success": True,
        "message": "Settings saved successfully"
    })
    response.set_cookie('api_key', new_api_key, max_age=30*24*60*60)  # 30 days
    
    # Handle file uploads if we have a valid RAG system
    indexed_files = []
    if rag_system is not None:
        files = request.files.getlist('files[]')
        
        for file in files:
            if file and file.filename:
                # Save uploaded file to tmp directory
                tmp_dir = os.path.join(os.getcwd(), 'tmp')
                os.makedirs(tmp_dir, exist_ok=True)
                file_path = os.path.join(tmp_dir, file.filename)
                file.save(file_path)
                
                # Index file in RAG system
                try:
                    rag_system.index_text_file(file_path)
                    indexed_files.append(file.filename)
                    logger.info(f"Indexed file: {file.filename}")
                except Exception as e:
                    logger.error(f"Error indexing file {file.filename}: {e}")
        
        # Update indexed files list in session data
        if 'indexed_files' not in session_data:
            session_data['indexed_files'] = []
        session_data['indexed_files'].extend(indexed_files)
        
        # Handle URLs
        urls = request.form.get('urls', '').splitlines()
        for url in urls:
            if url.strip():
                try:
                    article_info = article_manager.fetch_and_save_related_articles(
                        query=f"site:{url}", 
                        save_format="string",
                        time_limit_seconds=settings['timeout']
                    )
                    
                    # Index any found articles
                    for article in article_info:
                        if 'content' in article and article['content']:
                            # Add to RAG system as a text chunk
                            chunks = rag_system.text_processor.chunk_by_semantic_units(article['content'])
                            for i, chunk in enumerate(chunks):
                                # Generate embedding and add to vector store
                                embedding = rag_system.embedding_service.get_embedding(chunk)
                                rag_system.vector_store.add_texts(
                                    [chunk], 
                                    [embedding], 
                                    [{"source": article['url'], "chunk_id": i, "title": article['title']}]
                                )
                            
                            logger.info(f"Indexed article: {article['title']}")
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
    
    # Add indexed files to response
    response = jsonify({
        "success": True,
        "message": "Settings saved successfully",
        "indexed_files": indexed_files
    })
    response.set_cookie('api_key', new_api_key, max_age=30*24*60*60)  # 30 days
    
    return response

@app.route('/send_message', methods=['POST'])
def send_message():
    """Process a user message and return AI response"""
    global model, rag_system
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Check if model and RAG are initialized
    if model is None:
        return jsonify({"error": "Model not initialized. Please set up your API key."}), 400
    
    # Get or create chat session
    session_data = get_or_create_session(session.get('session_id'))
    session['session_id'] = session_data['id']
    
    # Add user message to history
    session_data['messages'].append({"role": "user", "content": user_message})
    
    # Determine if we should use RAG
    use_rag = True
    if user_message.lower().startswith('/chat'):
        use_rag = False
        user_message = user_message[5:].strip()  # Remove the /chat command
    
    # Process the message
    try:
        settings = session_data['settings']
        model.temperature = settings.get('temperature', DEFAULT_SETTINGS['temperature'])
        
        if use_rag and rag_system and rag_system.is_loaded():
            # Use RAG system if we have indexed content
            logger.info("Using RAG system for response")
            
            try:
                rag_response = rag_system.query(
                    question=user_message,
                    k=3,
                    model=settings.get('model', DEFAULT_SETTINGS['model'])
                )
                
                # Confirm rag_response is a dictionary
                logger.info(f"RAG response type: {type(rag_response)}")
                if isinstance(rag_response, dict):
                    logger.info(f"RAG response keys: {rag_response.keys()}")
                
            except Exception as rag_error:
                logger.error(f"Error in RAG query: {rag_error}")
                logger.error(f"Exception type: {type(rag_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            response = rag_response['answer']
            
            # Add source information if available
            if 'source_documents' in rag_response and rag_response['source_documents']:
                sources = []
                
                # Debug logging to see exactly what's in source_documents
                logger.info(f"Number of source documents: {len(rag_response['source_documents'])}")
                
                for i, doc in enumerate(rag_response['source_documents']):
                    try:
                        # Log the structure of each document
                        logger.info(f"Document {i} type: {type(doc)}")
                        
                        if isinstance(doc, dict):
                            logger.info(f"Document {i} keys: {doc.keys()}")
                            
                            if 'metadata' in doc:
                                logger.info(f"Document {i} metadata type: {type(doc['metadata'])}")
                                
                                if isinstance(doc['metadata'], dict):
                                    logger.info(f"Document {i} metadata keys: {doc['metadata'].keys()}")
                                    
                                    # Log each metadata item's type
                                    for key, value in doc['metadata'].items():
                                        logger.info(f"Document {i} metadata['{key}'] type: {type(value)}")
                                        if isinstance(value, list):
                                            logger.info(f"Document {i} metadata['{key}'] is a list with {len(value)} items")
                                    
                                    if 'source' in doc['metadata']:
                                        source = doc['metadata']['source']
                                        logger.info(f"Document {i} source value: {source}")
                                        logger.info(f"Document {i} source type: {type(source)}")
                                        
                                        # Handle different source types
                                        if isinstance(source, str):
                                            sources.append(source)
                                        elif isinstance(source, list):
                                            logger.info(f"Document {i} source is a list: {source}")
                                            # Add each item from the list
                                            for item in source:
                                                if isinstance(item, str):
                                                    sources.append(item)
                                                else:
                                                    logger.warning(f"Non-string item in source list: {item} (type: {type(item)})")
                                        else:
                                            # Convert other types to string
                                            sources.append(str(source))
                                            logger.info(f"Converted source to string: {str(source)}")
                    
                    except Exception as doc_error:
                        logger.error(f"Error processing document {i}: {doc_error}")
                        logger.error(f"Exception type: {type(doc_error)}")
                        logger.error(f"Document content: {str(doc)[:500]}...")  # First 500 chars
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Remove duplicates using dict.fromkeys()
                if sources:
                    unique_sources = list(dict.fromkeys(sources))
                    logger.info(f"Final unique sources: {unique_sources}")
                    response += "\n\nSources: " + ", ".join(unique_sources)
                else:
                    logger.info("No sources found in documents")
                    
        else:
            # Use standard model if no RAG content or explicitly requested
            logger.info("Using standard model for response")
            response = model.generate_response(
                user_message, 
                max_tokens=settings.get('max_tokens', DEFAULT_SETTINGS['max_tokens']),
                timeout=settings.get('timeout', DEFAULT_SETTINGS['timeout'])
            )
        
        # Add response to history
        session_data['messages'].append({"role": "assistant", "content": response})
        
        # Return the AI response
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a more detailed error for debugging
        error_details = {
            "error": f"Error processing message: {str(e)}",
            "type": str(type(e)),
            "message": user_message,
            "use_rag": use_rag,
            "stack_trace": traceback.format_exc()
        }
        return jsonify(error_details), 500
    
@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Create a new chat session"""
    # Create new session
    session_data = get_or_create_session()
    session['session_id'] = session_data['id']
    
    return jsonify({
        "success": True,
        "session_id": session_data['id']
    })

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Get the current chat history"""
    session_data = get_or_create_session(session.get('session_id'))
    session['session_id'] = session_data['id']
    
    return jsonify({
        "success": True,
        "messages": session_data['messages']
    })

@app.route('/get_chats', methods=['GET'])
def get_chats():
    """Get list of all chat sessions"""
    chats = []
    for chat_id, chat_data in CHAT_SESSIONS.items():
        # Extract a preview from the first few messages
        preview = ""
        for msg in chat_data['messages']:
            if msg['role'] == 'user':
                preview = msg['content']
                break
        
        if not preview and chat_data['messages']:
            preview = "New chat"
        
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

@app.route('/select_chat/<chat_id>', methods=['GET'])
def select_chat(chat_id):
    """Select a specific chat session"""
    if chat_id in CHAT_SESSIONS:
        session['session_id'] = chat_id
        return jsonify({
            "success": True,
            "messages": CHAT_SESSIONS[chat_id]['messages']
        })
    else:
        return jsonify({
            "success": False,
            "error": "Chat session not found"
        }), 404

@app.route('/system_info', methods=['GET'])
def system_info():
    """Get information about the system"""
    global rag_system
    
    rag_stats = {"status": "not loaded"}
    if rag_system and rag_system.is_loaded():
        rag_stats = rag_system.get_index_stats()
    
    return jsonify({
        "rag_system": rag_stats,
        "model_info": {
            "default_model": DEFAULT_SETTINGS['model'],
            "temperature": DEFAULT_SETTINGS['temperature'],
            "max_tokens": DEFAULT_SETTINGS['max_tokens']
        },
        "active_sessions": len(CHAT_SESSIONS)
    })

@app.route('/toggle', methods=['POST'])
def toggle():
    global toggle_state
    toggle_state = not toggle_state
    return jsonify({'state': toggle_state, 'text': 'On' if toggle_state else 'Off'})

@app.route('/get_state', methods=['GET'])
def get_state():
    return jsonify({'state': toggle_state, 'text': 'On' if toggle_state else 'Off'})

@app.route('/get_indexed_files', methods=['GET'])
def get_indexed_files():
    """Get a list of all indexed files"""
    try:
        # Get current session
        session_data = get_or_create_session(session.get('session_id'))
        session['session_id'] = session_data['id']
        
        # Get files from session data
        indexed_files = session_data.get('indexed_files', [])
        
        # Also check the tmp directory for additional files
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        if os.path.exists(tmp_dir):
            dir_files = os.listdir(tmp_dir)
            # Add any files that are not already in the session data
            for file in dir_files:
                if file not in indexed_files and os.path.isfile(os.path.join(tmp_dir, file)):
                    indexed_files.append(file)
        
        return jsonify({
            "success": True,
            "files": indexed_files
        })
    except Exception as e:
        logger.error(f"Error getting indexed files: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    """Delete a specific indexed file"""
    global rag_system
    
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                "success": False,
                "error": "Filename not provided"
            }), 400
        
        # Get current session
        session_data = get_or_create_session(session.get('session_id'))
        session['session_id'] = session_data['id']
        
        # Remove from session data
        if filename in session_data.get('indexed_files', []):
            session_data['indexed_files'].remove(filename)
        
        # Remove file from tmp directory
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        file_path = os.path.join(tmp_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {filename}")
        
        # Remove from vector store if RAG system is loaded
        if rag_system and rag_system.is_loaded():
            try:
                # Remove from vector store using the delete_file_from_index method
                rag_system.delete_file_from_index(filename)
                logger.info(f"Removed {filename} from vector store")
            except Exception as e:
                logger.warning(f"Could not remove {filename} from vector store: {e}")
        
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    """Delete all indexed files"""
    global rag_system
    
    try:
        # Get current session
        session_data = get_or_create_session(session.get('session_id'))
        session['session_id'] = session_data['id']
        
        # Clear session data
        session_data['indexed_files'] = []
        
        # Clear tmp directory
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        if os.path.exists(tmp_dir):
            # This will recreate the empty directory
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
        
        # Reset vector store if RAG system is loaded
        if rag_system and rag_system.is_loaded():
            try:
                # Clear the entire vector store
                rag_system.clear()
                logger.info("Cleared vector store")
            except Exception as e:
                logger.warning(f"Could not fully reset vector store: {e}")
        
        return jsonify({
            "success": True,
            "message": "All files deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting all files: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('articles', exist_ok=True)
    os.makedirs('vector_index', exist_ok=True)
    
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
    
    # Try to load existing vector index if available
    try:
        if os.path.exists('vector_index') and rag_system is not None:
            logger.info("Loading existing vector index...")
            rag_system.load_index('vector_index')
            logger.info(f"Vector index loaded: {rag_system.get_index_stats()}")
    except Exception as e:
        logger.error(f"Error loading vector index: {e}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
    # Ensure required directories exist
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('articles', exist_ok=True)
    os.makedirs('vector_index', exist_ok=True)
    
    # Try to load existing vector index if available
    try:
        if os.path.exists('vector_index') and rag_system is not None:
            logger.info("Loading existing vector index...")
            rag_system.load_index('vector_index')
            logger.info(f"Vector index loaded: {rag_system.get_index_stats()}")
    except Exception as e:
        logger.error(f"Error loading vector index: {e}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)