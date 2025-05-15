from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata for an academic document"""
    title: str
    authors: List[str]
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    
class DocumentChunk(BaseModel):
    """A chunk of text from a document with its metadata"""
    text: str
    document_id: str
    chunk_id: str
    section: Optional[str] = None
    page_num: Optional[int] = None

class DocumentSummary(BaseModel):
    """Summary of a document"""
    document_id: str
    summary_text: str
    summary_type: str = "general"  # general, methods, results, etc.
    length: Literal["short", "medium", "long"] = "medium"
