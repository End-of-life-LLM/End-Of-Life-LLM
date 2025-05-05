"""Utilities for processing text files."""

from typing import Any, Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

    
class Text_Processor:
    """Class for processing text files and evaluating retrieval performance."""
    
    @staticmethod
    def chunk_by_semantic_units(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 200) -> List[str]:
        """
        Split text into semantic chunks based on paragraph and sentence boundaries.
        
        Args:
            text: Text to chunk.
            max_chunk_size: Maximum characters per chunk.
            min_chunk_size: Minimum characters per chunk.
            
        Returns:
            List of semantically chunked text segments.
        """
        # Split by paragraphs first (double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph is too long, split into sentences
            if len(paragraph) > max_chunk_size:
                sentences = sent_tokenize(paragraph)
                
                for sentence in sentences:
                    # If adding this sentence exceeds the max size and we already have content,
                    # save the current chunk and start a new one
                    if len(current_chunk) + len(sentence) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                    else:
                        current_chunk += sentence + " "
            
            # If paragraph fits within limits or we're building a new chunk
            elif len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Save current chunk and start a new one with this paragraph
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def read_and_chunk_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Read and chunk a text file.
        
        Args:
            file_path: Path to the text file.
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            Tuple of (chunks, metadata).
        """
        # Read the text file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        
        # Split text into chunks
        chunks = Text_Processor.chunk_by_semantic_units(
            text, 
            max_chunk_size=chunk_size, 
            min_chunk_size=chunk_size // 4
        )
        
        # Create metadata for each chunk
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "source": file_path,
                "chunk_id": i,
                "start_char": None,  # Could be tracked if needed
                "end_char": None,    # Could be tracked if needed
            })
        
        return chunks, metadata