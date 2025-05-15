from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum

class QueryType(str, Enum):
    SUMMARIZE = "summarize"
    EXTRACT_INFO = "extract_info"
    ANSWER_QUESTION = "answer_question"
    COMPARE_PAPERS = "compare_papers"
    GENERATE_CITATION = "generate_citation"
    LITERATURE_REVIEW = "literature_review"
    
class AgentQuery(BaseModel):
    """A query to the research assistant"""
    query_type: QueryType
    query_text: str
    document_ids: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None