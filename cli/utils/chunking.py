import re
import uuid
from typing import List, Dict, Any, Optional


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 128) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) < chunk_size:
        return [text]
    
    # Clean and normalize the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position for this chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a good breaking point
        if end < text_length:
            # Try to break at paragraph, then sentence, then word boundary
            break_at = text.rfind('\n\n', start, end)
            if break_at == -1:
                break_at = text.rfind('. ', start, end)
            if break_at == -1:
                break_at = text.rfind(' ', start, end)
                
            # If we found a good breaking point, use it
            if break_at != -1:
                end = break_at + 1
        else:
            end = text_length
        
        # Add the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next position accounting for overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0
        
        # Avoid infinite loop
        if start >= end:
            break
    
    return chunks


def chunk_document(text: str, metadata: Dict[str, Any], 
                  chunk_size: int = 512, chunk_overlap: int = 128) -> List[Dict[str, Any]]:
    """
    Chunk a document text and attach metadata to each chunk
    
    Args:
        text: Document text to chunk
        metadata: Document metadata
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries with id, text, and metadata for each chunk
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    doc_id = metadata.get('id', str(uuid.uuid4()))
    
    result = []
    for i, chunk_text in enumerate(chunks):
        # Create unique ID for the chunk
        chunk_id = f"{doc_id}_{i}"
        
        # Create chunk with metadata
        chunk = {
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "parent_id": doc_id
            }
        }
        
        result.append(chunk)
    
    return result