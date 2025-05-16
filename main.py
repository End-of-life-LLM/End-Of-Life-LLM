"""
Main application file for the Cloud LLM Model system.
Runs a Flask web server to provide a chat interface.
"""

import os
import logging
import threading
import time
from flask import Flask, render_template, request, jsonify, session, redirect
from dotenv import load_dotenv
from Cloud_LLM_Model.RAG.controller import RAGController
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

fetch_process = {
    "active": False,
    "start_time": None,
    "end_time": None,
    "duration": 0,
    "count": 0,
    "embedded_count": 0,  # Track successfully embedded articles
    "embedding_failures": 0,  # Track embedding failures
    "search_query": "",
    "urls": [],
    "cancel_requested": False,
    "status": "idle",  # idle, running, completed, cancelled, error
    "message": "",  # Status message
    "can_embed": False,  # Whether embedding is possible
    "rag_was_enabled": False  # Track if RAG was enabled before fetch
}


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
@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():
    """Start the article fetching process with immediate embedding."""
    global controller, fetch_process
    
    # Check if already fetching
    if fetch_process["active"]:
        return jsonify({
            "success": False,
            "message": "A fetch process is already running. Please wait for it to complete or cancel it."
        })
    
    # Get form data
    search_query = request.form.get('searchQuery', '').strip()
    urls = request.form.get('urls', '').strip()
    fetch_duration = int(request.form.get('fetchDuration', 5))
    
    # Get max articles - use a much higher default than before
    max_articles = int(request.form.get('maxArticlesPerQuery', 100))
    
    # Validate inputs
    if not search_query and not urls:
        return jsonify({
            "success": False,
            "message": "Please provide either a search query or URLs to fetch articles."
        })
    
    # Parse URLs - don't limit the number of URLs to process
    url_list = []
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
    
    # Get current session - needed for file tracking
    session_id = request.form.get('session_id') or session.get('session_id')
    session_data = controller.get_or_create_session(session_id)
    
    # Check if API key is available for embedding
    if not controller.api_key:
        api_key = None
        # Try to get API key from session if available
        if 'settings' in session_data and 'apiKey' in session_data['settings']:
            api_key = session_data['settings']['apiKey']
            
        # If we found a key in the session, update the controller
        if api_key:
            logger.info("Using API key from session for embedding")
            controller.api_key = api_key
    
    # Ensure RAG controller is ready - use our new helper function
    can_embed = ensure_rag_controller_ready(controller, session_data)
    
    # Update controller's article manager max_results if needed
    if controller.article_controller:
        if controller.article_controller.searcher.max_results < max_articles:
            controller.article_controller.searcher.max_results = max_articles
            logger.info(f"Updated article manager max_results to {max_articles}")
    
    # Initialize fetch process
    fetch_process["active"] = True
    fetch_process["start_time"] = time.time()
    fetch_process["end_time"] = fetch_process["start_time"] + (fetch_duration * 60)
    fetch_process["duration"] = fetch_duration
    fetch_process["count"] = 0
    fetch_process["embedded_count"] = 0  # New counter for successfully embedded articles
    fetch_process["embedding_failures"] = 0  # New counter for embedding failures
    fetch_process["search_query"] = search_query
    fetch_process["urls"] = url_list
    fetch_process["cancel_requested"] = False
    fetch_process["status"] = "running"
    fetch_process["max_articles"] = max_articles
    fetch_process["session_id"] = session_id
    fetch_process["message"] = "Initializing..."
    
    fetch_process["can_embed"] = can_embed
    
    # If there are direct URLs, process them immediately
    if url_list:
        # Process URLs in a separate thread to avoid blocking
        threading.Thread(target=process_urls, args=(url_list, controller, request.form)).start()
        fetch_process["message"] = f"Processing {len(url_list)} URLs with direct embedding" if can_embed else f"Processing {len(url_list)} URLs (embedding unavailable)"
        
    # If there's a search query, start the timed search in a separate thread
    if search_query:
        threading.Thread(target=run_search_fetch, args=(search_query, fetch_duration, controller, request.form)).start()
        if fetch_process["message"] == "Initializing...":
            fetch_process["message"] = f"Running search for '{search_query}' with direct embedding" if can_embed else f"Running search for '{search_query}' (embedding unavailable)"
    
    # Return success
    return jsonify({
        "success": True,
        "status": "started",
        "message": fetch_process["message"],
        "duration": fetch_duration,
        "max_articles": max_articles,
        "embedding_enabled": can_embed,
        "urls_count": len(url_list) if url_list else 0
    })
def run_search_fetch(query, duration, controller, form_data):
    """Run the search-based article fetch process with direct embedding."""
    global fetch_process
    
    logger.info(f"Starting search fetch with direct embedding for query: {query}, duration: {duration} minutes")
    
    try:
        # Get current session
        session_id = form_data.get('session_id')
        session_data = controller.get_or_create_session(session_id)
        
        # Calculate end time
        end_time = time.time() + (duration * 60)
        
        # Maximum number of articles to fetch
        max_articles = fetch_process.get("max_articles", 100)
        
        # Set max_articles on the searcher object
        if controller.article_controller:
            controller.article_controller.searcher.max_results = max_articles
        
        # ENHANCED: Initialize RAG if needed for direct embedding
        if controller.rag_controller is None and controller.api_key:
            logger.info("Initializing RAG controller for direct embedding")
            controller.rag_controller = RAGController(
                api_key=controller.api_key,
                embedding_model="text-embedding-3-large",
                cache_enabled=True
            )
            # Load existing vector store if available
            controller._load_existing_vector_index()
        
        # Initialize counters for logging
        successfully_fetched = 0
        successfully_embedded = 0
        embedding_failures = 0
        
        # Continue fetching until time's up or cancelled
        while time.time() < end_time and not fetch_process["cancel_requested"]:
            try:
                # Use the article controller to fetch articles
                if controller.article_controller:
                    # Set time limit for this batch (remaining time or 60 seconds, whichever is less)
                    time_remaining = max(1, int(end_time - time.time()))
                    batch_time_limit = min(120, time_remaining)  # Use 120 seconds
                    
                    # Fetch articles - use save_format='file' to save to disk
                    results = controller.article_controller.fetch_and_save_related_articles(
                        query=query,
                        save_format='file',
                        time_limit_seconds=batch_time_limit
                    )
                    
                    # Process results
                    if results:
                        batch_size = len(results)
                        fetch_process["count"] += batch_size
                        successfully_fetched += batch_size
                        
                        # ENHANCED: Always try to index articles regardless of current RAG state
                        for article in results:
                            if 'filepath' in article:
                                try:
                                    # Add the file to the session's indexed files
                                    filename = os.path.basename(article['filepath'])
                                    if filename not in session_data.get('indexed_files', []):
                                        if 'indexed_files' not in session_data:
                                            session_data['indexed_files'] = []
                                        session_data['indexed_files'].append(filename)
                                    
                                    # Index the article if RAG is available
                                    if controller.rag_controller and controller.rag_controller.is_loaded():
                                        chunks = controller.rag_controller.index_text_file(article['filepath'])
                                        logger.info(f"✓ Successfully indexed {chunks} chunks from article: {article['title']}")
                                        successfully_embedded += 1
                                        fetch_process["embedded_count"] = fetch_process.get("embedded_count", 0) + 1
                                        
                                        # Enable RAG if not already enabled
                                        if not controller.rag_enabled:
                                            # Save original state if this is the first time
                                            if not fetch_process.get("rag_was_enabled", False):
                                                fetch_process["rag_was_enabled"] = controller.rag_enabled
                                            
                                            controller.rag_enabled = True
                                            logger.info("Automatically enabled RAG system after successful embedding")
                                    else:
                                        logger.warning(f"RAG controller not ready for embedding: {article['title']}")
                                        embedding_failures += 1
                                        fetch_process["embedding_failures"] = fetch_process.get("embedding_failures", 0) + 1
                                        
                                except Exception as e:
                                    logger.error(f"Error indexing article: {e}")
                                    import traceback
                                    logger.error(f"Traceback: {traceback.format_exc()}")
                                    embedding_failures += 1
                                    fetch_process["embedding_failures"] = fetch_process.get("embedding_failures", 0) + 1
                                    fetch_process["embedding_failures"] = fetch_process.get("embedding_failures", 0) + 1
                        
                        # Save the updated index if we have a RAG controller
                        if controller.rag_controller and controller.rag_controller.is_loaded():
                            controller.rag_controller.save_index('vector_index')
                        
                        # Save the updated session
                        controller.file_manager.save_chat_to_disk(session_data['id'], session_data)
                        
                        logger.info(f"Fetched {batch_size} articles, total: {fetch_process['count']}, embedded: {successfully_embedded}")
                    
                    # Check if we've reached the maximum limit
                    if fetch_process["count"] >= max_articles:
                        logger.info(f"Reached maximum number of articles ({max_articles})")
                        break
                    
                    # Check if the search has exhausted all results
                    if controller.article_controller.searcher.result_found >= controller.article_controller.searcher.max_results:
                        logger.info("Search has exhausted all available results")
                        break
                        
                else:
                    logger.warning("Article controller not initialized")
                    break
                
                # Add a small delay between fetches to avoid rate limiting
                time.sleep(3)  # 3 seconds between fetches
                
            except Exception as e:
                logger.error(f"Error in search fetch iteration: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(5)  # Wait a bit longer after an error
        
        # Mark as completed
        finish_fetch_process("completed" if not fetch_process["cancel_requested"] else "cancelled")
        
        logger.info(f"Search fetch completed with status: {fetch_process['status']}, fetched {fetch_process['count']} articles, successfully embedded: {successfully_embedded}, embedding failures: {embedding_failures}")
        
        # Save the session one final time to ensure all indexed files are saved
        controller.file_manager.save_chat_to_disk(session_data['id'], session_data)
        
    except Exception as e:
        logger.error(f"Error in run_search_fetch: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        finish_fetch_process("error")      

def process_urls(url_list, controller, form_data):
    """Process the given URLs to fetch articles and embed them directly."""
    global fetch_process
    
    logger.info(f"Processing {len(url_list)} URLs with direct embedding")
    
    try:
        # Get current session
        session_id = form_data.get('session_id')
        session_data = controller.get_or_create_session(session_id)
        
        # Initialize counters for logging
        successfully_processed = 0
        embedding_failures = 0
        
        for i, url in enumerate(url_list):
            if fetch_process["cancel_requested"]:
                logger.info("URL processing cancelled")
                break
                
            logger.info(f"Processing URL {i+1}/{len(url_list)}: {url}")
            
            try:
                # Use the article controller to fetch the article content
                if controller.article_controller:
                    # Fetch the article directly using the WebArticleSearcher's fetch_article method
                    content = controller.article_controller.searcher.fetch_article(url)
                    
                    if content:
                        # Extract a title from the URL if needed
                        title = url.split('/')[-1].replace('-', ' ').replace('_', ' ')
                        if '.' in title:
                            title = title.split('.')[0]
                        
                        # Save the article
                        filepath = controller.article_controller.saver.save_article(
                            title=title,
                            content=content,
                            url=url,
                            format='file'
                        )
                        
                        if filepath:
                            # Add the file to the session's indexed files
                            filename = os.path.basename(filepath)
                            if filename not in session_data.get('indexed_files', []):
                                if 'indexed_files' not in session_data:
                                    session_data['indexed_files'] = []
                                session_data['indexed_files'].append(filename)
                            
                            # ENHANCED: Direct embedding - always try to index regardless of RAG state
                            try:
                                # Ensure RAG controller is ready
                                rag_ready = ensure_rag_controller_ready(controller, session_data)
                                
                                if rag_ready:
                                    chunks = controller.rag_controller.index_text_file(filepath)
                                    logger.info(f"✓ Successfully indexed {chunks} chunks from URL: {url}")
                                    
                                    # Save the index after each file for robustness
                                    controller.rag_controller.save_index('vector_index')
                                    
                                    # Enable RAG if not already enabled
                                    if not controller.rag_enabled:
                                        # Save original state if this is the first time
                                        if not fetch_process.get("rag_was_enabled", False):
                                            fetch_process["rag_was_enabled"] = controller.rag_enabled
                                        
                                        controller.rag_enabled = True
                                        logger.info("Automatically enabled RAG system after successful embedding")
                                    
                                    successfully_processed += 1
                                    fetch_process["embedded_count"] = fetch_process.get("embedded_count", 0) + 1
                                else:
                                    logger.warning(f"RAG controller not ready for embedding: {url}")
                                    embedding_failures += 1
                                    fetch_process["embedding_failures"] = fetch_process.get("embedding_failures", 0) + 1
                                    
                            except Exception as e:
                                logger.error(f"Error indexing URL content: {e}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                embedding_failures += 1
                            
                            # Increment count
                            fetch_process["count"] += 1
                            
                            # Save the updated session periodically
                            if i % 5 == 0 or i == len(url_list) - 1:  # Every 5 URLs or the last one
                                controller.file_manager.save_chat_to_disk(session_data['id'], session_data)
                        else:
                            logger.warning(f"Failed to save article from URL: {url}")
                    else:
                        logger.warning(f"No content fetched from URL: {url}")
                else:
                    logger.warning("Article controller not initialized")
                    break
                
                # Add a small delay between URLs to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Completed processing {len(url_list)} URLs. Successfully embedded: {successfully_processed}, Embedding failures: {embedding_failures}")
        
        # Save the session one final time to ensure all indexed files are saved
        controller.file_manager.save_chat_to_disk(session_data['id'], session_data)
        
        # If no search query and all URLs processed, mark as completed
        if not fetch_process["search_query"]:
            finish_fetch_process("completed")
            
    except Exception as e:
        logger.error(f"Error in process_urls: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        finish_fetch_process("error")

@app.route('/fetch_status', methods=['GET'])
def fetch_status():
    """Get the status of the current fetch process with embedding details."""
    global fetch_process
    
    # Calculate additional information
    remaining_time = max(0, int(fetch_process["end_time"] - time.time())) if fetch_process["end_time"] else 0
    elapsed_time = int(time.time() - fetch_process["start_time"]) if fetch_process["start_time"] else 0
    
    # Calculate embedding percentage if applicable
    embedding_percentage = 0
    if fetch_process.get("count", 0) > 0:
        embedding_percentage = round((fetch_process.get("embedded_count", 0) / fetch_process.get("count", 1)) * 100)
    
    # Generate a detailed status message
    status_message = fetch_process.get("message", "")
    if fetch_process["active"]:
        if fetch_process["status"] == "running":
            status_message = (
                f"Fetched {fetch_process['count']} articles"
                f"{f', embedded {fetch_process.get('embedded_count', 0)} ({embedding_percentage}%)' if fetch_process.get('can_embed', False) else ''}"
                f"{f', {remaining_time} seconds remaining' if remaining_time > 0 else ''}"
            )
        elif fetch_process["status"] == "cancelling":
            status_message = "Cancelling fetch process..."
    else:
        if fetch_process["status"] == "completed":
            status_message = (
                f"Completed. Fetched {fetch_process['count']} articles"
                f"{f', embedded {fetch_process.get('embedded_count', 0)} ({embedding_percentage}%)' if fetch_process.get('can_embed', False) else ''}"
            )
        elif fetch_process["status"] == "cancelled":
            status_message = (
                f"Cancelled. Fetched {fetch_process['count']} articles"
                f"{f', embedded {fetch_process.get('embedded_count', 0)} ({embedding_percentage}%)' if fetch_process.get('can_embed', False) else ''}"
            )
        elif fetch_process["status"] == "error":
            status_message = "Error occurred during fetch process"
    
    # Return current status with enhanced information
    return jsonify({
        "success": True,
        "active": fetch_process["active"],
        "status": fetch_process["status"],
        "count": fetch_process["count"],
        "embedded_count": fetch_process.get("embedded_count", 0),
        "embedding_failures": fetch_process.get("embedding_failures", 0),
        "embedding_percentage": embedding_percentage,
        "start_time": fetch_process["start_time"],
        "end_time": fetch_process["end_time"],
        "elapsed_time": elapsed_time,
        "duration": fetch_process["duration"],
        "remaining": remaining_time,
        "can_embed": fetch_process.get("can_embed", False),
        "message": status_message,
        "rag_enabled": controller.rag_enabled if hasattr(controller, "rag_enabled") else False
    })
@app.route('/cancel_fetch', methods=['POST'])
def cancel_fetch():
    """Cancel the current fetch process"""
    global fetch_process
    
    # Check if there's an active fetch
    if not fetch_process["active"]:
        return jsonify({
            "success": False,
            "message": "No active fetch process to cancel."
        })
    
    # Set cancel flag
    fetch_process["cancel_requested"] = True
    fetch_process["status"] = "cancelling"
    fetch_process["message"] = "Cancellation requested. This may take a moment to complete."
    
    # Return success
    return jsonify({
        "success": True,
        "message": fetch_process["message"]
    })
def finish_fetch_process(status="completed"):
    """Properly finish the fetch process, updating status and restoring settings if needed."""
    global fetch_process, controller
    
    # Update status
    fetch_process["status"] = status
    fetch_process["active"] = False
    
    # Create appropriate status message
    embedding_percentage = 0
    if fetch_process.get("count", 0) > 0:
        embedding_percentage = round((fetch_process.get("embedded_count", 0) / fetch_process.get("count", 1)) * 100)
    
    if status == "completed":
        fetch_process["message"] = (
            f"Completed. Fetched {fetch_process['count']} articles"
            f"{f', embedded {fetch_process.get('embedded_count', 0)} ({embedding_percentage}%)' if fetch_process.get('can_embed', False) else ''}"
        )
    elif status == "cancelled":
        fetch_process["message"] = (
            f"Cancelled. Fetched {fetch_process['count']} articles"
            f"{f', embedded {fetch_process.get('embedded_count', 0)} ({embedding_percentage}%)' if fetch_process.get('can_embed', False) else ''}"
        )
    elif status == "error":
        fetch_process["message"] = f"Error occurred during fetch process. Fetched {fetch_process['count']} articles."
    
    # Log completion
    logger.info(f"Fetch process {status} with {fetch_process['count']} articles fetched, {fetch_process.get('embedded_count', 0)} embedded")
    
    # Save index one final time if we have a RAG controller and articles were embedded
    if controller.rag_controller and controller.rag_controller.is_loaded() and fetch_process.get("embedded_count", 0) > 0:
        try:
            controller.rag_controller.save_index('vector_index')
            logger.info("Final index save completed")
        except Exception as e:
            logger.error(f"Error during final index save: {e}")
    
    # Option to restore previous RAG state
    # Uncomment the following if you want to restore the previous RAG state
    """
    if fetch_process.get("embedded_count", 0) > 0 and "rag_was_enabled" in fetch_process:
        # We only restore the previous state if the setting was automatically changed
        # and if RAG was originally disabled
        original_state = fetch_process.get("rag_was_enabled", True)
        if not original_state and controller.rag_enabled:
            # Don't automatically disable RAG if we embedded articles
            # Let the user make that decision
            logger.info("Keeping RAG enabled after embedding articles")
    """
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