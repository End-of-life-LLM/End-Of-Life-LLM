import json
import os
import pickle
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

class Retrieval:
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        if not self.api_key:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        
        self.index = None
        self.dimension = 1536
        self.texts = []
        self.metadata = []
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API with retry logic."""
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        return response["data"][0]["embedding"]

    def load_index(self, directory: str = "vector_index"):
        """Load the index, texts, and metadata from disk."""
        index_path = os.path.join(directory, "faiss_index.bin")
        texts_path = os.path.join(directory, "texts.pkl")
        metadata_path = os.path.join(directory, "metadata.json")
        
        if not (os.path.exists(index_path) and os.path.exists(texts_path) and os.path.exists(metadata_path)):
            print(f"Missing index files in {directory}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load texts
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.dimension = self.index.d
        
        print(f"Index loaded from {directory} with {len(self.texts)} documents")
        return True
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for most similar documents to a query."""
        if self.index is None or not self.texts:
            return []
        
        # Get query embedding
        query_embedding = np.array([self.get_embedding(query)]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.texts):  # Safety check
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(1 / (1 + distances[0][i]))  # Convert distance to similarity score
                })
        
        return results

    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self.index is not None and len(self.texts) > 0

    def get_index_stats(self) -> Dict[str, Any]:
        """Return statistics about the loaded index."""
        if not self.is_loaded():
            return {
                "status": "not_loaded",
                "document_count": 0
            }
        
        # Count documents by source
        sources = {}
        for meta in self.metadata:
            source = meta.get("source", "unknown")
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        return {
            "status": "loaded",
            "document_count": len(self.texts),
            "dimension": self.dimension,
            "sources": sources
        }

    def search_with_metadata_filter(self, query: str, filter_key: str, filter_value: Any, k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents with a metadata filter."""
        # First get more results than needed
        initial_results = self.search(query, k=k*3)
        
        # Filter by metadata
        filtered_results = [r for r in initial_results if r["metadata"].get(filter_key) == filter_value]
        
        # Return top k after filtering
        return filtered_results[:k]

    def get_document_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by its chunk ID."""
        for i, meta in enumerate(self.metadata):
            if meta.get("chunk_id") == chunk_id:
                return {
                    "text": self.texts[i],
                    "metadata": meta
                }
        return None

    def get_context_window(self, chunk_id: int, window_size: int = 1) -> List[Dict[str, Any]]:
        """Get surrounding context for a specific chunk."""
        # Find the document first
        doc = self.get_document_by_id(chunk_id)
        if not doc:
            return []
        
        source = doc["metadata"].get("source")
        curr_id = chunk_id
        
        # Find all chunks from the same source
        source_chunks = []
        for i, meta in enumerate(self.metadata):
            if meta.get("source") == source:
                source_chunks.append({
                    "index": i,
                    "chunk_id": meta.get("chunk_id"),
                    "text": self.texts[i],
                    "metadata": meta
                })
        
        # Sort by chunk_id
        source_chunks.sort(key=lambda x: x["chunk_id"])
        
        # Find the current chunk position
        current_pos = next((i for i, chunk in enumerate(source_chunks) if chunk["chunk_id"] == curr_id), -1)
        if current_pos == -1:
            return []
        
        # Get the window
        start = max(0, current_pos - window_size)
        end = min(len(source_chunks), current_pos + window_size + 1)
        
        return [{"text": c["text"], "metadata": c["metadata"]} for c in source_chunks[start:end]]
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        if self.index is None or not self.texts:
            return {"answer": "No indexed documents. Please index some documents first."}
        
        # Search for relevant documents
        search_results = self.search(question, k)
        
        if not search_results:
            return {"answer": "No relevant documents found."}
        
        # Extract texts from search results
        context_texts = [result["text"] for result in search_results]
        
        # Generate answer
        answer = self.generate_answer(question, context_texts)
        
        return {
            "answer": answer,
            "source_documents": search_results
        }
