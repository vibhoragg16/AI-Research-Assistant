from langgraph.graph import StateGraph, END

from ..models.agent import AgentState
from ..tools.agent_tools import AgentTools
from . import nodes
from . import router

def setup_graph(tools: AgentTools):
    """Create and return the workflow graph"""

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route", lambda state: router.route(state, tools))
    workflow.add_node("process_documents", lambda state: nodes.process_documents(state, tools))
    workflow.add_node("retrieve_information", lambda state: nodes.retrieve_information(state, tools))
    workflow.add_node("generate_summary", lambda state: nodes.generate_summary(state, tools))
    workflow.add_node("extract_methodology", lambda state: nodes.extract_methodology(state, tools))
    workflow.add_node("extract_claims", lambda state: nodes.extract_claims(state, tools))
    workflow.add_node("compare_documents", lambda state: nodes.compare_documents(state, tools))
    workflow.add_node("generate_citation", lambda state: nodes.generate_citation(state, tools))
    workflow.add_node("answer_question", lambda state: nodes.answer_question(state, tools))
    workflow.add_node("generate_literature_review", lambda state: nodes.generate_literature_review(state, tools))
    workflow.add_node("provide_final_answer", lambda state: nodes.provide_final_answer(state, tools))
    
    # Add edges from router to all other nodes
    workflow.add_edge("route", "process_documents")
    workflow.add_edge("route", "retrieve_information")
    workflow.add_edge("route", "generate_summary")
    workflow.add_edge("route", "extract_methodology")
    workflow.add_edge("route", "extract_claims")
    workflow.add_edge("route", "compare_documents")
    workflow.add_edge("route", "generate_citation")
    workflow.add_edge("route", "answer_question")
    workflow.add_edge("route", "generate_literature_review")
    workflow.add_edge("route", "provide_final_answer")
    
    # Add direct progression edges
    workflow.add_edge("process_documents", "retrieve_information")
    workflow.add_edge("retrieve_information", "generate_summary")
    workflow.add_edge("generate_summary", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("extract_methodology", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("extract_claims", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("compare_documents", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("generate_citation", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("answer_question", "provide_final_answer")  # Go directly to final answer
    workflow.add_edge("generate_literature_review", "provide_final_answer")  # Go directly to final answer
    
    # Final answer leads to the end
    workflow.add_edge("provide_final_answer", END)
    
    # Set the entry point
    workflow.set_entry_point("route")
    
    return workflow.compile()

