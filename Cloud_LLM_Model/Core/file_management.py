"""
File Management Module for the Cloud LLM Model system.
Handles file operations, chat persistence, and API key storage.
"""

import os
import time
import json
import logging
import shutil
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FileManager")

# Default settings (same as in Controller)
DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30,
    "model": "gpt-4.1"
}

class FileManager:
    """
    Handles all file-related operations for the Cloud LLM Model system.
    Responsible for:
    - Managing chat files (save/load/delete)
    - Handling API key storage
    - Managing temporary files
    - Directory structure maintenance
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the FileManager.
        
        Args:
            base_dir: Optional base directory. If None, uses current working directory.
        """
        # Set up base directories
        self.base_dir = base_dir or os.getcwd()
        self.chats_dir = os.path.join(self.base_dir, "chats")
        self.tmp_dir = os.path.join(self.base_dir, "tmp")
        
        # Ensure required directories exist
        self._ensure_directories()
        
        # Initialize state
        self.chat_sessions = {}
        
        # Load existing chats from disk
        self._load_chats_from_disk()
        
        logger.info("FileManager initialization complete")
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.chats_dir, exist_ok=True)
        logger.debug("Required directories created or verified")
    
    #----------------------------------------------------------------------
    # Chat session management methods
    #----------------------------------------------------------------------
    
    def _load_chats_from_disk(self) -> None:
        """
        Load existing chat sessions from disk.
        """
        try:
            if not os.path.exists(self.chats_dir):
                logger.info("Chat directory does not exist, creating it")
                os.makedirs(self.chats_dir, exist_ok=True)
                return
                
            # Read each chat file in the chats directory
            chat_files = [f for f in os.listdir(self.chats_dir) if f.endswith('.json')]
            logger.info(f"Found {len(chat_files)} chat files on disk")
            
            loaded_count = 0
            for chat_file in chat_files:
                try:
                    file_path = os.path.join(self.chats_dir, chat_file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        
                    # Ensure required fields exist
                    chat_id = chat_data.get('id', chat_file.replace('.json', ''))
                    chat_data['id'] = chat_id
                    
                    if 'created_at' not in chat_data:
                        chat_data['created_at'] = time.time()
                        
                    if 'messages' not in chat_data:
                        chat_data['messages'] = []
                        
                    if 'settings' not in chat_data:
                        chat_data['settings'] = DEFAULT_SETTINGS.copy()
                        
                    if 'indexed_files' not in chat_data:
                        chat_data['indexed_files'] = []
                        
                    # Add to in-memory chat sessions
                    self.chat_sessions[chat_id] = chat_data
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading chat file {chat_file}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully loaded {loaded_count} chat sessions from disk")
            
        except Exception as e:
            logger.error(f"Error loading chats from disk: {str(e)}")
    
    def save_chat_to_disk(self, chat_id: str, chat_data: Dict[str, Any]) -> bool:
        """
        Save a chat session to disk.
        
        Args:
            chat_id: ID of the chat to save
            chat_data: The chat data to save
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            if not chat_id:
                logger.warning("No chat ID provided, cannot save to disk")
                return False
                
            # Ensure the chat directory exists
            os.makedirs(self.chats_dir, exist_ok=True)
            
            # Save the chat data to a JSON file
            file_path = os.path.join(self.chats_dir, f"{chat_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
                
            # Update in-memory cache
            self.chat_sessions[chat_id] = chat_data
                
            logger.debug(f"Saved chat {chat_id} to disk")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chat {chat_id} to disk: {str(e)}")
            return False
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID to retrieve
            
        Returns:
            Dict containing session data
        """
        if session_id and session_id in self.chat_sessions:
            return self.chat_sessions[session_id]
        
        # Create new session
        new_id = session_id or str(uuid.uuid4())
        session_data = {
            "id": new_id,
            "created_at": time.time(),
            "messages": [],
            "settings": DEFAULT_SETTINGS.copy(),
            "indexed_files": [],
        }
        
        # Save the new session to disk
        self.save_chat_to_disk(new_id, session_data)
        logger.info(f"Created new chat session with ID: {new_id}")
        
        return session_data
        
    def delete_chat(self, chat_id: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete a chat session.
        
        Args:
            chat_id: ID of chat to delete
            
        Returns:
            Dict with status or error tuple
        """
        try:
            if not chat_id:
                return {
                    "success": False,
                    "error": "Chat ID not provided"
                }, 400
                
            # Check if chat exists
            if chat_id not in self.chat_sessions:
                return {
                    "success": False,
                    "error": "Chat not found"
                }, 404
                
            # Remove from memory
            self.chat_sessions.pop(chat_id)
            
            # Remove file from disk
            file_path = os.path.join(self.chats_dir, f"{chat_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted chat file: {chat_id}.json")
                
            return {
                "success": True,
                "message": f"Chat '{chat_id}' deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def delete_tmp_file(self, filename: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete a specific file from the tmp directory.
        
        Args:
            filename: Name of file to delete
            
        Returns:
            Dict with status or error tuple
        """
        try:
            if not filename:
                return {
                    "success": False,
                    "error": "Filename not provided"
                }, 400
            
            # Remove file from tmp directory
            file_path = os.path.join(self.tmp_dir, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {filename}")
                return {
                    "success": True,
                    "message": f"File '{filename}' deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"File '{filename}' not found"
                }, 404
                
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def clear_tmp_directory(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Clear all files from the tmp directory.
        
        Returns:
            Dict with status or error tuple
        """
        try:
            # Clear tmp directory
            if os.path.exists(self.tmp_dir):
                # This will recreate the empty directory
                shutil.rmtree(self.tmp_dir)
                os.makedirs(self.tmp_dir, exist_ok=True)
                logger.info("Cleared tmp directory")
            
            return {
                "success": True,
                "message": "All temporary files deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error clearing tmp directory: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def clear_all_chats(self) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Delete all chat sessions.
        
        Returns:
            Dict with status or error tuple
        """
        try:
            # Clear in-memory chats
            self.chat_sessions = {}
            
            # Clear chat files from disk
            if os.path.exists(self.chats_dir):
                chat_files = [f for f in os.listdir(self.chats_dir) if f.endswith('.json')]
                for chat_file in chat_files:
                    try:
                        file_path = os.path.join(self.chats_dir, chat_file)
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting chat file {chat_file}: {str(e)}")
                        
            return {
                "success": True,
                "message": "All chats deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error clearing all chats: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    def rename_chat(self, chat_id: str, new_title: str) -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
        """
        Rename a chat session.
        
        Args:
            chat_id: ID of chat to rename
            new_title: New title for the chat
            
        Returns:
            Dict with status or error tuple
        """
        try:
            if not chat_id:
                return {
                    "success": False,
                    "error": "Chat ID not provided"
                }, 400
                
            if not new_title:
                return {
                    "success": False,
                    "error": "New title not provided"
                }, 400
                
            # Check if chat exists
            if chat_id not in self.chat_sessions:
                return {
                    "success": False,
                    "error": "Chat not found"
                }, 404
                
            # Update chat title
            chat_data = self.chat_sessions[chat_id] 
            chat_data['title'] = new_title
            
            # Save to disk
            self.save_chat_to_disk(chat_id, chat_data)
            logger.info(f"Chat {chat_id} renamed to '{new_title}'")
            
            return {
                "success": True,
                "message": f"Chat renamed successfully"
            }
        except Exception as e:
            logger.error(f"Error renaming chat {chat_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }, 500
    
    #----------------------------------------------------------------------
    # API key management methods
    #----------------------------------------------------------------------
    
    def save_api_key_to_env(self, api_key: str) -> bool:
        """
        Save API key to .env file for persistence across restarts.
        
        Args:
            api_key: API key to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            env_file_path = os.path.join(self.base_dir, '.env')
            
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
                        env_lines[i] = f'OPENAI_API_KEY={api_key}\n'
                        key_exists = True
                        break
                
                # Append key if it doesn't exist
                if not key_exists:
                    env_lines.append(f'OPENAI_API_KEY={api_key}\n')
                
                # Write updated content back to .env file
                with open(env_file_path, 'w') as f:
                    f.writelines(env_lines)
            else:
                # Create new .env file
                with open(env_file_path, 'w') as f:
                    f.write(f'OPENAI_API_KEY={api_key}\n')
            
            logger.info("API key saved to .env file")
            return True
        except Exception as e:
            logger.error(f"Error saving API key to .env file: {str(e)}")
            return False
    
    def load_api_key_from_env(self) -> Optional[str]:
        """
        Load API key from .env file.
        
        Returns:
            API key if found, None otherwise
        """
        try:
            env_file_path = os.path.join(self.base_dir, '.env')
            
            if not os.path.exists(env_file_path):
                logger.info(".env file not found")
                return None
                
            # Read .env file
            with open(env_file_path, 'r') as f:
                env_lines = f.readlines()
            
            # Look for OPENAI_API_KEY
            for line in env_lines:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    if api_key:
                        logger.info("API key loaded from .env file")
                        return api_key
            
            logger.info("API key not found in .env file")
            return None
        except Exception as e:
            logger.error(f"Error loading API key from .env file: {str(e)}")
            return None
    
    #----------------------------------------------------------------------
    # File utility methods
    #----------------------------------------------------------------------
    
    def save_uploaded_file(self, file_obj: Any, filename: Optional[str] = None) -> Union[str, None]:
        """
        Save an uploaded file to the tmp directory.
        
        Args:
            file_obj: The file object to save
            filename: Optional filename to use instead of the original name
            
        Returns:
            The saved file path if successful, None otherwise
        """
        try:
            if not file_obj:
                logger.warning("No file provided")
                return None
                
            # Determine filename
            actual_filename = filename or file_obj.filename
            if not actual_filename:
                logger.warning("No filename provided")
                return None
                
            # Ensure tmp directory exists
            os.makedirs(self.tmp_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(self.tmp_dir, actual_filename)
            file_obj.save(file_path)
            logger.info(f"Saved uploaded file: {actual_filename}")
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return None
    
    def get_file_list(self) -> List[str]:
        """
        Get a list of all files in the tmp directory.
        
        Returns:
            List of filenames
        """
        try:
            if not os.path.exists(self.tmp_dir):
                return []
                
            # Get all files in the tmp directory
            files = [f for f in os.listdir(self.tmp_dir) 
                    if os.path.isfile(os.path.join(self.tmp_dir, f))]
            
            return files
        except Exception as e:
            logger.error(f"Error getting file list: {str(e)}")
            return []
    
    def get_chat_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all chat sessions.
        
        Returns:
            List of chat session data (with minimal info)
        """
        try:
            # Return a simplified list of chats with just the essential info
            chat_list = []
            for chat_id, chat_data in self.chat_sessions.items():
                chat_list.append({
                    "id": chat_id,
                    "title": chat_data.get("title", f"Chat {chat_id[:8]}"),
                    "created_at": chat_data.get("created_at", 0),
                    "message_count": len(chat_data.get("messages", [])),
                })
            
            # Sort by creation time (newest first)
            chat_list.sort(key=lambda x: x["created_at"], reverse=True)
            
            return chat_list
        except Exception as e:
            logger.error(f"Error getting chat list: {str(e)}")
            return []
        

    def save_indexed_files(self, session_id: str, file_list: List[str]) -> bool:
        """
        Save the list of indexed files for a session.
        
        Args:
            session_id: ID of the session
            file_list: List of indexed file names
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not session_id:
                logger.warning("No session ID provided, cannot save indexed files")
                return False
                
            session_data = self.get_or_create_session(session_id)
            
            # Update indexed files list
            session_data['indexed_files'] = file_list
            
            # Save to disk
            self.save_chat_to_disk(session_id, session_data)
            logger.info(f"Saved {len(file_list)} indexed files for session {session_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving indexed files: {str(e)}")
            return False
        
    def save_indexed_files(self, session_id: str, file_list: List[str]) -> bool:
        """
        Save the list of indexed files for a session.
        
        Args:
            session_id: ID of the session
            file_list: List of indexed file names
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not session_id:
                logger.warning("No session ID provided, cannot save indexed files")
                return False
                
            session_data = self.get_or_create_session(session_id)
            
            # Update indexed files list
            session_data['indexed_files'] = file_list
            
            # Save to disk
            self.save_chat_to_disk(session_id, session_data)
            logger.info(f"Saved {len(file_list)} indexed files for session {session_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving indexed files: {str(e)}")
            return False
