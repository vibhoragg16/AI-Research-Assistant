from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class ResearchClaim(BaseModel):
    """A research claim extracted from a paper"""
    claim: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    document_id: str
    location: Optional[str] = None

class ComparisonResult(BaseModel):
    """Comparison between two or more papers"""
    similarities: List[str]
    differences: List[str]
    methodology_comparison: Optional[str] = None
    result_comparison: Optional[str] = None
    document_ids: List[str]

class Citation(BaseModel):
    """A formatted citation"""
    document_id: str
    citation_text: str
    style: str = "APA"  # Default to APA style

class MethodologyInfo(BaseModel):
    """Information about research methodology"""
    approach: str
    datasets: List[str] = []
    algorithms: List[str] = []
    evaluation_metrics: List[str] = []
    limitations: List[str] = []
    document_id: str

