"""
Main controller for the Cloud LLM Model system.
"""

import os
import time
import json
import logging
import uuid
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Controller")

# Import components
from Cloud_LLM_Model.Core.model import Model
from Cloud_LLM_Model.Core.supporting_model import Supporting_Model
from Cloud_LLM_Model.RAG.controller import RAGController
from WebSearshing.webArticleManger import WebArticleManager  # Fixed typo in import
from Cloud_LLM_Model.Utils.token_counter import Token_Counter
from Cloud_LLM_Model.Core.file_management import FileManager  # Import our new FileManager module

# Default settings
DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "model": "gpt-4.1"
}

class Controller:
    """
    Main controller class for the Cloud LLM Model system.
    Integrates the Model, Supporting_Model, and RAG subsystems.
    Uses FileManager for file operations and chat persistence.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Controller.
        
        Args:
            api_key: Optional OpenAI API key. If None, will try to load from environment.
        """
        # Set up base directories
        self.base_dir = os.getcwd()
        self.vector_index_dir = os.path.join(self.base_dir, "vector_index")
        self.articles_dir = os.path.join(self.base_dir, "articles")
        
        # Initialize the FileManager
        self.file_manager = FileManager(base_dir=self.base_dir)
        
        # Initialize state variables
        self.api_key = api_key or self.file_manager.load_api_key_from_env()
        self.rag_enabled = False
        
        # Initialize components
        self.model = None
        self.supporting_model = None
        self.rag_controller = None
        self.article_controller = None
        self.token_counter = Token_Counter()
        
        # Initialize components if API key is available
        if self.api_key:
            self.initialize_components(self.api_key)
            
        # Try to load existing vector index if available
        self._load_existing_vector_index()
        
        logger.info("Controller initialization complete")
    
    #----------------------------------------------------------------------
    # Directory and initialization methods
    #----------------------------------------------------------------------
    
    def initialize_components(self, api_key: str) -> bool:
        """
        Initialize or reinitialize all components with the given API key.
        
        Args:
            api_key: OpenAI API key to use for components
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing components with API key")
            self.api_key = api_key
            
            # Save API key to .env file using FileManager
            self.file_manager.save_api_key_to_env(api_key)
            
            # Initialize main model with temperature from settings
            self.model = Model(api_key=api_key)
            
            # Initialize supporting model
            self.supporting_model = Supporting_Model(api_key=api_key)
            
            # Initialize RAG controller with embedding model and cache
            self.rag_controller = RAGController(
                api_key=api_key, 
                embedding_model="text-embedding-3-large", 
                cache_enabled=True
            )
            
            # Initialize article controller for web searches
            self.article_controller = WebArticleManager(
                max_results=5, 
                save_directory=self.articles_dir
            )
            
            logger.info("All components successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            # Log stack trace for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_existing_vector_index(self) -> bool:
        """
        Load existing vector index if available.
        
        Returns:
            bool: True if index loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.vector_index_dir) and self.rag_controller is not None:
                logger.info("Loading existing vector index...")
                self.rag_controller.load_index(self.vector_index_dir)
                stats = self.rag_controller.get_index_stats()
                logger.info(f"Vector index loaded: {stats}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vector index: {str(e)}")
            return False
    
    #----------------------------------------------------------------------
    # Chat session management methods (now using FileManager)
    #----------------------------------------------------------------------
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get an existing session or create a new one using the FileManager.
        
        Args:
            session_id: Optional session ID to retrieve
            
        Returns:
            Dict containing session data
        """
        return self.file_manager.get_or_create_session(session_id)
        
    def delete_chat(self, chat_id: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete a chat session using the FileManager.
        
        Args:
            chat_id: ID of chat to delete
            
        Returns:
            Dict with status or error tuple
        """
        return self.file_manager.delete_chat(chat_id)
    
    def delete_file(self, filename: str, session_id: Optional[str] = None) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete a specific indexed file.
        
        Args:
            filename: Name of file to delete
            session_id: Optional session ID
            
        Returns:
            Dict with status or error tuple
        """
        try:
            if not filename:
                return {
                    "success": False,
                    "error": "Filename not provided"
                }, 400
            
            # Get current session
            session_data = self.get_or_create_session(session_id)
            
            # Remove from session data
            if filename in session_data.get('indexed_files', []):
                session_data['indexed_files'].remove(filename)
                
                # Save chat to disk after updating
                self.file_manager.save_chat_to_disk(session_data['id'], session_data)
            
            # Delete the file from tmp directory using FileManager
            self.file_manager.delete_tmp_file(filename)
            
            # Remove from vector store if RAG system is loaded
            if self.rag_controller and self.rag_controller.is_loaded():
                try:
                    # Remove from vector store using the delete_file_from_index method
                    self.rag_controller.delete_file_from_index(filename)
                    logger.info(f"Removed {filename} from vector store")
                except Exception as e:
                    logger.warning(f"Could not remove {filename} from vector store: {str(e)}")
            
            return {
                "success": True,
                "message": f"File '{filename}' deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def delete_all_files(self, session_id: Optional[str] = None) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete all indexed files.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Dict with status or error tuple
        """
        try:
            # Get current session
            session_data = self.get_or_create_session(session_id)
            
            # Clear session data
            session_data['indexed_files'] = []
            
            # Save chat to disk after updating
            self.file_manager.save_chat_to_disk(session_data['id'], session_data)
            
            # Clear tmp directory using FileManager
            self.file_manager.clear_tmp_directory()
            
            # Reset vector store if RAG system is loaded
            if self.rag_controller and self.rag_controller.is_loaded():
                try:
                    # Clear the entire vector store
                    self.rag_controller.clear()
                    logger.info("Cleared vector store")
                except Exception as e:
                    logger.warning(f"Could not fully reset vector store: {str(e)}")
            
            return {
                "success": True,
                "message": "All files deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting all files: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def clear_all_chats(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete all chat sessions using the FileManager.
        
        Returns:
            Dict with status or error tuple
        """
        return self.file_manager.clear_all_chats()
    
    def rename_chat(self, chat_id: str, new_title: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Rename a chat session using the FileManager.
        
        Args:
            chat_id: ID of chat to rename
            new_title: New title for the chat
            
        Returns:
            Dict with status or error tuple
        """
        return self.file_manager.rename_chat(chat_id, new_title)
    
    #----------------------------------------------------------------------
    # Message processing methods
    #----------------------------------------------------------------------
    
    def process_message(self, user_message: str, session_id: Optional[str] = None) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Process a user message and return AI response.
        
        Args:
            user_message: The message from the user
            session_id: Optional session ID
            
        Returns:
            Dict with response or error tuple (dict, status_code)
        """
        if not user_message:
            return {"error": "No message provided"}, 400
        
        # Check if model is initialized
        if self.model is None:
            return {"error": "Model not initialized. Please set up your API key."}, 400
        
        # Get or create chat session using FileManager
        session_data = self.get_or_create_session(session_id)
        
        # Add user message to history
        session_data['messages'].append({"role": "user", "content": user_message})
        
        # Determine if we should use RAG based on toggle state and command prefix
        use_rag = self.rag_enabled
        processed_message = user_message
        
        # Check for /chat prefix to bypass RAG
        if user_message.lower().startswith('/chat'):
            use_rag = False
            processed_message = user_message[5:].strip()  # Remove the /chat command
        
        # Process the message
        try:
            # Get settings from session
            settings = session_data['settings']
            
            # Configure model temperature
            self.model.temperature = settings.get('temperature', DEFAULT_SETTINGS['temperature'])
            
            # Determine whether to use RAG or standard model
            if use_rag and self.rag_controller and self.rag_controller.is_loaded():
                response = self._process_with_rag(processed_message, settings)
            else:
                response = self._process_with_standard_model(processed_message, settings)
            
            # Add response to history
            session_data['messages'].append({"role": "assistant", "content": response})
            
            # Save chat to disk after updating using FileManager
            self.file_manager.save_chat_to_disk(session_data['id'], session_data)
            
            # Return the AI response
            return {"response": response}
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return a more detailed error for debugging
            return {
                "error": f"Error processing message: {str(e)}",
                "type": str(type(e)),
                "message": user_message,
                "use_rag": use_rag
            }, 500
        
    def _process_with_standard_model(self, message: str, settings: Dict[str, Any]) -> str:
        """
        Process a message using the standard model without RAG.
        
        Args:
            message: User message to process
            settings: Current settings dictionary
            
        Returns:
            Response text from standard model
        """
        logger.info("Using standard model for response (RAG is disabled or not available)")
        
        # Generate response using the standard model
        response = self.model.generate_response(
            message, 
            max_tokens=settings.get('max_tokens', DEFAULT_SETTINGS['max_tokens']),
            timeout=settings.get('timeout', DEFAULT_SETTINGS['timeout'])
        )
        
        return response
    
    def _process_with_rag(self, message: str, settings: Dict[str, Any]) -> str:
        """
        Process a message using the RAG system.
        
        Args:
            message: User message to process
            settings: Current settings dictionary
            
        Returns:
            Response text from RAG system
        """
        logger.info("Using RAG system for response")
        
        # Query the RAG system
        rag_response = self.rag_controller.query(
            question=message,
            k=3,
            model=settings.get('model', DEFAULT_SETTINGS['model'])
        )
        
        # Get the answer from the response
        response = rag_response['answer']
        
        # Add source information if available
        response = self._add_source_information(response, rag_response)
        
        return response
    

    def _add_source_information(self, response: str, rag_response: Dict[str, Any]) -> str:
        """
        Add source information to the response if available.
        
        Args:
            response: The original response text
            rag_response: The RAG system response data
            
        Returns:
            Response text with source information appended
        """
        if 'source_documents' not in rag_response or not rag_response['source_documents']:
            return response
            
        sources = []
        logger.debug(f"Processing {len(rag_response['source_documents'])} source documents")
        
        for doc in rag_response['source_documents']:
            try:
                if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
                    metadata = doc['metadata']
                    
                    if 'source' in metadata:
                        source = metadata['source']
                        
                        # Handle different source types
                        if isinstance(source, str):
                            sources.append(source)
                        elif isinstance(source, list):
                            # Add each item from the list
                            sources.extend([str(item) for item in source if item])
                        else:
                            # Convert other types to string
                            sources.append(str(source))
            
            except Exception as e:
                logger.error(f"Error processing source document: {str(e)}")
                continue
        
        # Remove duplicates using dict.fromkeys() and add to response
        if sources:
            unique_sources = list(dict.fromkeys(sources))
            response += "\n\nSources: " + ", ".join(unique_sources)
        
        return response
    

    def get_rag_enabled(self) -> Dict[str, bool]:
        """
        Get the current state of the RAG system.
        
        Returns:
            Dict with state information
        """
        return {
            "state": self.rag_enabled
        }

    def set_rag_enabled(self, state: bool) -> Dict[str, Any]:
        """
        Set the RAG system enabled/disabled state.
        
        Args:
            state: New state (True for enabled, False for disabled)
            
        Returns:
            Dict with state information
        """
        self.rag_enabled = state
        logger.info(f"RAG system {'enabled' if state else 'disabled'}")
        return {
            "state": self.rag_enabled,
            "message": f"RAG system {'enabled' if state else 'disabled'}"
        }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system.
        
        Returns:
            Dict with system information
        """
        has_api_key = bool(self.api_key)
        rag_loaded = False
        
        # Check if RAG controller is initialized and index is loaded
        if self.rag_controller is not None:
            rag_loaded = self.rag_controller.is_loaded()
        
        # Get index stats if available
        index_stats = {}
        if rag_loaded and self.rag_controller:
            try:
                index_stats = self.rag_controller.get_index_stats()
            except Exception as e:
                logger.error(f"Error getting index stats: {str(e)}")
        
        return {
            "api_key_set": has_api_key,
            "model_initialized": self.model is not None,
            "supporting_model_initialized": self.supporting_model is not None,
            "rag_controller_initialized": self.rag_controller is not None,
            "rag_index_loaded": rag_loaded,
            "index_stats": index_stats,
            "version": "1.0.0"  # Add version information
        }

    def get_indexed_files(self, session_id: Optional[str] = None) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Get a list of all indexed files for the current session.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Dict with file information or error tuple
        """
        try:
            # Get current session
            session_data = self.get_or_create_session(session_id)
            
            # Get indexed files from session data
            indexed_files = session_data.get('indexed_files', [])
            
            # If RAG controller is loaded, get additional file information
            file_details = []
            if self.rag_controller and self.rag_controller.is_loaded():
                # Get stats for each file
                index_stats = self.rag_controller.get_index_stats()
                
                # For each indexed file, add available stats
                for filename in indexed_files:
                    # Count chunks associated with this file
                    chunk_count = 0
                    if hasattr(self.rag_controller.vector_store, 'metadata'):
                        for metadata in self.rag_controller.vector_store.metadata:
                            if metadata.get("source", "") == filename:
                                chunk_count += 1
                    
                    file_details.append({
                        "filename": filename,
                        "chunks": chunk_count,
                        "indexed": True
                    })
            else:
                # Just return basic information if RAG is not available
                file_details = [{"filename": f, "indexed": True} for f in indexed_files]
            
            return {
                "success": True,
                "files": file_details
            }
        except Exception as e:
            logger.error(f"Error getting indexed files: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
        

    def save_api_settings(self, form_data: Dict[str, Any], uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        Save API settings, handle file uploads, and process article searches/URLs.
        
        Args:
            form_data: Dictionary containing form data
            uploaded_files: List of uploaded file objects
            
        Returns:
            Dict with result information
        """
        try:
            # Extract session_id from form_data
            session_id = form_data.get('session_id')
            
            # Get the session data
            session_data = self.get_or_create_session(session_id)
            
            # Update API key and initialize components if key is provided
            api_key = form_data.get('apiKey')
            if api_key:
                # Save the API key for session
                session_data['settings']['apiKey'] = api_key
                
                # Update the API key and initialize components
                success = self.initialize_components(api_key)
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to initialize components with provided API key"
                    }
            
            # Update other settings
            if 'temperature' in form_data and form_data['temperature']:
                session_data['settings']['temperature'] = float(form_data['temperature'])
                
            if 'maxTokens' in form_data and form_data['maxTokens']:
                session_data['settings']['max_tokens'] = int(form_data['maxTokens'])
                
            if 'timeout' in form_data and form_data['timeout']:
                session_data['settings']['timeout'] = int(form_data['timeout'])
            
            # Process uploaded files if any
            result_files = []
            if uploaded_files:
                # Handle file uploads - limit to the specified number if provided
                max_files = int(form_data.get('numberOfFiles', 5))
                files_to_process = uploaded_files[:max_files]
                
                for file in files_to_process:
                    try:
                        if not file.filename:
                            continue
                            
                        # Save file using FileManager
                        file_path = self.file_manager.save_uploaded_file(file)
                        
                        if not file_path:
                            continue
                        
                        # Add to session's indexed files list if not already there
                        if file.filename not in session_data.get('indexed_files', []):
                            if 'indexed_files' not in session_data:
                                session_data['indexed_files'] = []
                            session_data['indexed_files'].append(file.filename)
                        
                        # Index the file if RAG controller is available
                        if self.rag_controller:
                            try:
                                # Check file type and use appropriate method
                                file_extension = os.path.splitext(file.filename)[1].lower()
                                
                                if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                                    # Text files
                                    chunks = self.rag_controller.index_text_file(file_path)
                                    logger.info(f"Indexed {chunks} chunks from {file.filename}")
                                else:
                                    # Unsupported file type
                                    logger.warning(f"Unsupported file type for {file.filename}")
                                
                                # Save the updated index
                                self.rag_controller.save_index('vector_index')
                                
                                result_files.append({
                                    "filename": file.filename,
                                    "success": True
                                })
                            except Exception as e:
                                logger.error(f"Error indexing file {file.filename}: {str(e)}")
                                result_files.append({
                                    "filename": file.filename,
                                    "success": False,
                                    "error": str(e)
                                })
                        else:
                            # RAG controller not available, just save the file reference
                            result_files.append({
                                "filename": file.filename,
                                "success": True,
                                "warning": "File saved but not indexed (RAG not initialized)"
                            })
                    except Exception as e:
                        logger.error(f"Error processing file {file.filename if hasattr(file, 'filename') else 'unknown'}: {str(e)}")
                        result_files.append({
                            "filename": file.filename if hasattr(file, 'filename') else "unknown",
                            "success": False,
                            "error": str(e)
                        })
            
            # Process direct URLs
            direct_urls = form_data.get('direct_urls', [])
            result_urls = []
            
            if direct_urls and self.article_controller:
                for url in direct_urls:
                    try:
                        # Use the article controller to fetch the article
                        article = self.article_controller.fetch_article(url)
                        
                        if article and article.get('text'):
                            # Create a filename for the article
                            clean_url = url.replace('://', '_').replace('/', '_')[:50]
                            filename = f"url_{int(time.time())}_{clean_url}.txt"
                            file_path = os.path.join(self.file_manager.tmp_dir, filename)
                            
                            # Write article content to file
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(f"Title: {article.get('title', 'Unknown')}\n\n")
                                f.write(f"URL: {url}\n\n")
                                f.write(article.get('text', ''))
                            
                            # Add to session's indexed files list
                            if filename not in session_data.get('indexed_files', []):
                                if 'indexed_files' not in session_data:
                                    session_data['indexed_files'] = []
                                session_data['indexed_files'].append(filename)
                            
                            # Index the article text if RAG controller is available
                            if self.rag_controller:
                                try:
                                    chunks = self.rag_controller.index_text_file(file_path)
                                    logger.info(f"Indexed {chunks} chunks from URL {url}")
                                    
                                    # Save the updated index
                                    self.rag_controller.save_index('vector_index')
                                    
                                    result_urls.append({
                                        "url": url,
                                        "success": True,
                                        "filename": filename,
                                        "chunks": chunks
                                    })
                                except Exception as e:
                                    logger.error(f"Error indexing URL {url}: {str(e)}")
                                    result_urls.append({
                                        "url": url,
                                        "success": False,
                                        "error": str(e)
                                    })
                            else:
                                # RAG controller not available, just save the file reference
                                result_urls.append({
                                    "url": url,
                                    "success": True,
                                    "filename": filename,
                                    "warning": "URL content saved but not indexed (RAG not initialized)"
                                })
                        else:
                            logger.warning(f"Failed to fetch article from URL: {url}")
                            result_urls.append({
                                "url": url,
                                "success": False,
                                "error": "Failed to fetch article"
                            })
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {str(e)}")
                        result_urls.append({
                            "url": url,
                            "success": False,
                            "error": str(e)
                        })
            
            # Process search queries
            search_queries = form_data.get('search_queries', [])
            search_results = []
            
            if search_queries and self.article_controller:
                max_articles = int(form_data.get('max_articles_per_query', 5))
                timeout = int(form_data.get('search_timeout', 30))
                
                for query in search_queries:
                    try:
                        # Fetch and save related articles
                        logger.info(f"Searching for articles related to: {query}")
                        
                        # Use fetch_and_save_related_articles method
                        articles = self.article_controller.fetch_and_save_related_articles(
                            query=query,
                            save_format='string',  # Get content as strings for processing
                            time_limit_seconds=timeout
                        )
                        
                        query_results = []
                        
                        # Process each found article
                        for article in articles:
                            try:
                                # Skip articles without content
                                if not article.get('content'):
                                    continue
                                    
                                # Create a clean filename
                                article_title = article.get('title', 'Unknown').replace(' ', '_')[:50]
                                filename = f"search_{int(time.time())}_{article_title}.txt"
                                file_path = os.path.join(self.file_manager.tmp_dir, filename)
                                
                                # Write article content to file
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(f"Title: {article.get('title', 'Unknown')}\n\n")
                                    f.write(f"URL: {article.get('url', 'Unknown')}\n\n")
                                    f.write(f"Search Query: {query}\n\n")
                                    f.write(article.get('content', ''))
                                
                                # Add to session's indexed files list
                                if filename not in session_data.get('indexed_files', []):
                                    if 'indexed_files' not in session_data:
                                        session_data['indexed_files'] = []
                                    session_data['indexed_files'].append(filename)
                                
                                # Index with RAG if available
                                if self.rag_controller:
                                    chunks = self.rag_controller.index_text_file(file_path)
                                    logger.info(f"Indexed {chunks} chunks from article: {article.get('title', 'Unknown')}")
                                    
                                    # Save the updated index
                                    self.rag_controller.save_index('vector_index')
                                    
                                    query_results.append({
                                        "title": article.get('title', 'Unknown'),
                                        "url": article.get('url', 'Unknown'),
                                        "success": True,
                                        "filename": filename,
                                        "chunks": chunks
                                    })
                                else:
                                    # RAG not available
                                    query_results.append({
                                        "title": article.get('title', 'Unknown'),
                                        "url": article.get('url', 'Unknown'),
                                        "success": True,
                                        "filename": filename,
                                        "warning": "Article saved but not indexed (RAG not initialized)"
                                    })
                            except Exception as e:
                                logger.error(f"Error processing article: {str(e)}")
                                query_results.append({
                                    "title": article.get('title', 'Unknown'),
                                    "url": article.get('url', 'Unknown'),
                                    "success": False, 
                                    "error": str(e)
                                })
                        
                        # Add query results to search results
                        search_results.append({
                            "query": query,
                            "articles_found": len(articles),
                            "articles_processed": len(query_results),
                            "results": query_results
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing search query '{query}': {str(e)}")
                        search_results.append({
                            "query": query,
                            "success": False,
                            "error": str(e)
                        })
            
            # Save the updated session data
            self.file_manager.save_chat_to_disk(session_data['id'], session_data)
            
            # Return success response
            return {
                "success": True,
                "session_id": session_data['id'],
                "files": result_files,
                "direct_urls": result_urls,
                "search_queries": search_results,
                "settings_saved": True
            }
            
        except Exception as e:
            logger.error(f"Error saving API settings: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }