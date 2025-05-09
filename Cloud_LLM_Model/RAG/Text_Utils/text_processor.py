"""Utilities for processing text files with improved chunking."""

from typing import Any, Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import re
from functools import lru_cache

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
class Text_Processor:
    """Class for processing text files and evaluating retrieval performance."""
    
    @staticmethod
    def chunk_by_semantic_units(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 200) -> List[str]:
        """
        Split text into semantic chunks based on paragraph and sentence boundaries.
        Uses improved semantic chunking that respects document structure.
        
        Args:
            text: Text to chunk.
            max_chunk_size: Maximum characters per chunk.
            min_chunk_size: Minimum characters per chunk.
            
        Returns:
            List of semantically chunked text segments.
        """
        # Detect if text has code blocks or structured content
        has_code = "```" in text or re.search(r'(def|class|import|function|var|const|let)\s+\w+', text) is not None
        
        # Adjust chunk size based on content
        if has_code:
            # Keep code blocks intact by using larger chunks
            effective_max_size = max(max_chunk_size, 1500)
            # Ensure we don't break code blocks in the middle
            chunks = Text_Processor._chunk_code_aware(text, effective_max_size, min_chunk_size)
        else:
            # Split by meaningful semantic boundaries
            chunks = Text_Processor._chunk_regular_text(text, max_chunk_size, min_chunk_size)
        
        # Post-process chunks to avoid tiny fragments
        return Text_Processor._merge_small_chunks(chunks, min_chunk_size)
    
    @staticmethod
    def _chunk_code_aware(text: str, max_size: int, min_size: int) -> List[str]:
        """
        Chunk text with awareness of code blocks to avoid breaking them.
        
        Args:
            text: Text to chunk.
            max_size: Maximum characters per chunk.
            min_size: Minimum characters per chunk.
            
        Returns:
            List of code-aware chunked text segments.
        """
        # Split on triple backticks to identify code blocks
        parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # If this is a code block
            if part.startswith('```') and part.endswith('```'):
                # If adding this code block would exceed max size and we already have content
                if len(current_chunk) + len(part) > max_size and len(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk += part
            else:
                # For non-code text, chunk by paragraphs
                paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
                
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) + 2 <= max_size:
                        if current_chunk and not current_chunk.endswith('\n\n'):
                            current_chunk += "\n\n"
                        current_chunk += paragraph
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = paragraph
                        else:
                            # If a single paragraph exceeds max size, split by sentences
                            current_chunk = Text_Processor._split_long_paragraph(paragraph, max_size, min_size, chunks)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    @staticmethod
    def _chunk_regular_text(text: str, max_size: int, min_size: int) -> List[str]:
        """
        Chunk regular text by semantic units like paragraphs and sentences.
        
        Args:
            text: Text to chunk.
            max_size: Maximum characters per chunk.
            min_size: Minimum characters per chunk.
            
        Returns:
            List of semantically chunked text segments.
        """
        # Split by paragraphs first (respect double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check for section headers to use as chunk boundaries
            is_header = re.match(r'^#+\s+', paragraph) is not None or (
                len(paragraph) < 100 and paragraph.upper() == paragraph
            )
            
            # Start new chunk on headers unless it would be too small
            if is_header and len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
                continue
                
            # If paragraph fits within limits or we're building a new chunk
            if len(current_chunk) + len(paragraph) + 2 <= max_size:
                if current_chunk and not current_chunk.endswith('\n\n'):
                    current_chunk += "\n\n"
                current_chunk += paragraph
            else:
                # Save current chunk if it's not too small
                if len(current_chunk) >= min_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Split long paragraph by sentences
                    current_chunk = Text_Processor._split_long_paragraph(paragraph, max_size, min_size, chunks, current_chunk)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def _split_long_paragraph(paragraph: str, max_size: int, min_size: int, 
                              chunks: List[str], current_text: str = "") -> str:
        """
        Split a long paragraph into sentence-based chunks.
        
        Args:
            paragraph: Long paragraph to split.
            max_size: Maximum chunk size.
            min_size: Minimum chunk size.
            chunks: List to append complete chunks to.
            current_text: Text to prepend to the first chunk.
            
        Returns:
            Any remaining text after chunking.
        """
        # If the paragraph is too long, split into sentences
        sentences = sent_tokenize(paragraph)
        
        # Start with any existing text
        current_chunk = current_text
        
        for sentence in sentences:
            # If adding this sentence would exceed max size and we have enough content
            if len(current_chunk) + len(sentence) > max_size and len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        
        return current_chunk
    
    @staticmethod
    def _merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
        """
        Merge chunks that are too small with adjacent chunks.
        
        Args:
            chunks: List of text chunks.
            min_size: Minimum chunk size.
            
        Returns:
            List of merged chunks.
        """
        if not chunks:
            return []
            
        result = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            # If current chunk is too small, merge it with the next one
            if len(current) < min_size:
                current += "\n\n" + chunks[i]
            # If next chunk is too small, merge it with current
            elif len(chunks[i]) < min_size:
                current += "\n\n" + chunks[i]
            else:
                # Both chunks are large enough, add current to result and move on
                result.append(current)
                current = chunks[i]
        
        # Add the last chunk
        if current:
            result.append(current)
            
        return result
    
    @staticmethod
    def estimate_optimal_chunk_size(text: str) -> int:
        """
        Estimate optimal chunk size based on text characteristics.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Suggested chunk size.
        """
        # Check if code-heavy
        code_pattern = re.compile(r'(def|class|import|function|var|const|let)\s+\w+')
        code_matches = len(code_pattern.findall(text))
        
        # Check if table/structured data-heavy
        table_indicators = text.count('|') + text.count('\t')
        
        # Check for long sentences
        sentences = sent_tokenize(text)
        avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
        
        # Calculate base chunk size
        if code_matches > 5:
            # Code-heavy documents need larger chunks
            base_size = 1500
        elif table_indicators > 20:
            # Documents with tables need medium-large chunks
            base_size = 1200
        elif avg_sentence_length > 100:
            # Academic/legal texts with long sentences
            base_size = 1000
        else:
            # Normal prose
            base_size = 800
            
        return base_size
    
    @staticmethod
    @lru_cache(maxsize=32)
    def read_and_chunk_file(file_path: str, chunk_size: int = None, chunk_overlap: int = 200) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Read and chunk a text file with dynamic chunk sizing.
        
        Args:
            file_path: Path to the text file.
            chunk_size: Maximum size of each chunk, or None for auto-sizing.
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
                
        # Determine file type based on extension
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        is_code = file_ext in ['py', 'js', 'java', 'cpp', 'c', 'cs', 'go', 'rb', 'php', 'ts', 'sh']
        
        # Auto-determine chunk size if not specified
        if chunk_size is None:
            # Use different defaults based on content type
            if is_code:
                chunk_size = 1500  # Larger chunks for code
            else:
                # Analyze text to determine optimal chunk size
                chunk_size = Text_Processor.estimate_optimal_chunk_size(text)
        
        # Adjust overlap for very small chunk sizes
        if chunk_size < 500:
            chunk_overlap = min(chunk_overlap, chunk_size // 4)
            
        # Split text into chunks
        chunks = Text_Processor.chunk_by_semantic_units(
            text, 
            max_chunk_size=chunk_size, 
            min_chunk_size=chunk_size // 4
        )
        
        # Create metadata for each chunk
        metadata = []
        chunk_start = 0
        
        for i, chunk in enumerate(chunks):
            # Calculate approx. char position in original text
            chunk_start = text.find(chunk, max(0, chunk_start - chunk_overlap))
            chunk_end = chunk_start + len(chunk)
            
            metadata.append({
                "source": file_path,
                "chunk_id": i,
                "start_char": chunk_start,
                "end_char": chunk_end,
                "is_code": is_code,
                "file_type": file_ext,
            })
            
            # Update for next chunk
            chunk_start = chunk_end
        
        return chunks, metadata