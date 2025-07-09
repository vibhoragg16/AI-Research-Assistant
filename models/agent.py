from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
#from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
#from .query import AgentQuery

class AgentAction(BaseModel):
    """Next action for the agent to take"""
    action: str
    action_input: Optional[Dict[str, Any]] = None
    
class AgentState(BaseModel):
    """State maintained during the agent's execution"""
    query: Dict[str, Any]
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    documents: Dict[str, Any] = Field(default_factory=dict)
    extracted_info: Dict[str, Any] = Field(default_factory=dict)
    next_actions: List[AgentAction] = Field(default_factory=list)
    current_action: Optional[AgentAction] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    step_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True