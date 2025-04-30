import json
import os
import pickle
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Embedding:
    def __init__(self, api_key: Optional[str] = None, embedding_model: str= "text-embedding-ada-002"):
        self.api_key = api_key
        if not self.api_key:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model


        self.index = None 
        self.dimension = 1536 
        self.text = []
        self.metadata = []

    ### After waiting between 1 and 60 seconds and max attempts of 6 this creates a vector from the string 
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API with retry logic."""
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        return response["data"][0]["embedding"]
    

    ### Using numpy for fast calculations of array
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        all_embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            all_embeddings.append(embedding)
        return np.array(all_embeddings).astype('float32')   
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = min(start + chunk_size, len(text))
            
            # If we're not at the end of the text, try to find a good break point
            if end < len(text):
                # Try to find a period, question mark, or exclamation point
                for i in range(min(chunk_overlap, end - start)):
                    if text[end - i - 1] in ['.', '!', '?', '\n'] and text[end - i:end - i + 1].isspace():
                        end = end - i
                        break
            
            # Add the chunk to our list
            chunks.append(text[start:end])
            
            # Move the start point, accounting for overlap
            start = end - chunk_overlap if end < len(text) else end
        
        return chunks
      
    def index_text_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Index a text file by splitting, embedding, and storing in FAISS."""
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into chunks
        chunks = self.chunk_text(text, chunk_size, chunk_overlap)
        
        # Create metadata for each chunk
        chunk_metadata = [
            {
                "source": file_path,
                "chunk_id": i,
                "start_char": None,  # Could track this if needed
                "end_char": None     # Could track this if needed
            }
            for i in range(len(chunks))
        ]
        
        # Get embeddings
        embeddings = self.get_embeddings(chunks)
        
        # Initialize FAISS index if not already done
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store original texts and metadata
        start_id = len(self.texts)
        self.texts.extend(chunks)
        self.metadata.extend(chunk_metadata)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        print(f"Indexed {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def save_index(self, directory: str = "vector_index"):
        """Save the index, texts, and metadata to disk."""
        if self.index is None:
            print("No index to save. Please index documents first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Save texts and metadata
        with open(os.path.join(directory, "texts.pkl"), 'wb') as f:
            pickle.dump(self.texts, f)
        
        with open(os.path.join(directory, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f)
        
        print(f"Index saved to {directory}")

    def get_index_info(self) -> Dict[str, Any]:
        """Return information about the current index."""
        if self.index is None:
            return {
                "status": "not_initialized",
                "document_count": 0
            }
        
        return {
            "status": "initialized",
            "document_count": len(self.texts),
            "dimension": self.dimension,
            "index_type": type(self.index).__name__
        }

    def is_indexed(self) -> bool:
        """Check if any documents have been indexed."""
        return self.index is not None and len(self.texts) > 0

    def clear_index(self) -> None:
        """Clear the current index, texts, and metadata."""
        self.index = None
        self.texts = []
        self.metadata = []
        print("Index cleared")

    def get_indexing_status(self, file_path: str) -> Dict[str, Any]:
        """Check if a specific file has been indexed by looking at metadata."""
        if not self.metadata:
            return {"indexed": False, "chunks": 0}
        
        # Check if file exists in metadata
        indexed_chunks = [m for m in self.metadata if m.get("source") == file_path]
        return {
            "indexed": len(indexed_chunks) > 0,
            "chunks": len(indexed_chunks)
        }

    def batch_index_files(self, file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Index multiple files at once and return statistics."""
        results = {}
        total_chunks = 0
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                results[file_path] = {"status": "error", "message": "File not found"}
                continue
            
            try:
                chunks = self.index_text_file(file_path, chunk_size, chunk_overlap)
                results[file_path] = {"status": "success", "chunks": chunks}
                total_chunks += chunks
            except Exception as e:
                results[file_path] = {"status": "error", "message": str(e)}
        
        return {"files": results, "total_chunks": total_chunks}