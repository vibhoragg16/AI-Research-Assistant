from ..models.agent import AgentState, AgentAction
from ..tools.agent_tools import AgentTools

#from llm.utils import format_message
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage


def create_router_prompt():
    """
    Creates the prompt template for the conversation router.
    
    This router is responsible for determining the best next step in a conversation
    based on the user's input and conversation history.
    
    Returns:
        ChatPromptTemplate: The formatted router prompt template
    """
    router_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are the controller for a research assistant agent system.
        Your job is to determine which action should be taken next based on the current query and state.
        Choose from the following actions:
        - process_documents: Process new documents (if document IDs are not in the state)
        - retrieve_information: Retrieve relevant information from documents
        - generate_summary: Generate a summary of documents
        - extract_methodology: Extract methodology information
        - extract_claims: Extract key claims
        - compare_documents: Compare multiple documents
        - generate_citation: Generate citations for documents
        - answer_question: Answer a specific question about documents
        - generate_literature_review: Generate a literature review
        - final_answer: Provide a final answer to the user query
       
        If all relevant actions have been completed, select 'final_answer' to finish the workflow.
        You must not repeat actions that have already been completed unless necessary."""),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="""Based on the above conversation and current state, what should be the next action?
        Current query: {query}
        Current state: {current_state}
       
        Respond with just the action name from the list above.""")
    ])
    
    return router_prompt

def route(state: AgentState, tools: AgentTools) -> Dict[str, Any]:
    """
    Routes the conversation to the appropriate next step based on the user's input.
    
    Args:
        state: The current state of the agent
        tools: The tools available to the agent
    
    Returns:
        Dict[str, Any]: Updated state and the next action to take
    """
    # Initialize state fields if not present
    if not hasattr(state, "step_count"):
        state.step_count = 0
    if not hasattr(state, "documents"):
        state.documents = {}
    if not hasattr(state, "extracted_info"):
        state.extracted_info = {}
    if not hasattr(state, "messages"):
        state.messages = []
    
    # Increment step count
    state.step_count += 1

    # Hard stop after 10 steps
    if state.step_count > 10:
        print("Step limit reached, forcing provide_final_answer")
        state.current_action = AgentAction(action="provide_final_answer")
        return {"state": state, "next": "provide_final_answer"}
    
    if len(state.messages) > 10:  # or another threshold
        print("Forcing final_answer for debugging")
        state.current_action = AgentAction(action="final_answer")
        return {"state": state, "next": "final_answer"}
    
    # Check if we need to process documents
    if not state.documents:
        print("No documents found, routing to process_documents")
        state.current_action = AgentAction(action="process_documents")
        return {"state": state, "next": "process_documents"}
    
    # Check if we've already processed documents in this step
    if state.current_action and state.current_action.action == "process_documents":
        print("Documents already processed in this step, moving to next action")
        state.current_action = AgentAction(action="retrieve_information")
        return {"state": state, "next": "retrieve_information"}
    
    # Check if we have any unprocessed documents
    has_unprocessed = False
    for doc_id, doc_info in state.documents.items():
        if not doc_info.get("processed", False):
            has_unprocessed = True
            break
    
    if has_unprocessed:
        print("Found unprocessed documents, routing to process_documents")
        state.current_action = AgentAction(action="process_documents")
        return {"state": state, "next": "process_documents"}
    
    # Check if we have retrieved information
    if "retrieved_chunks" in state.extracted_info:
        print("Information already retrieved, checking next action")
        
        # If we have retrieved information but no summaries, go to generate_summary
        if "summaries" not in state.extracted_info:
            print("No summaries found, routing to generate_summary")
            state.current_action = AgentAction(action="generate_summary")
            return {"state": state, "next": "generate_summary"}
    
    # If we have processed documents but no information retrieved yet
    if state.documents and "retrieved_chunks" not in state.extracted_info:
        print("Documents processed but no information retrieved yet, routing to retrieve_information")
        state.current_action = AgentAction(action="retrieve_information")
        return {"state": state, "next": "retrieve_information"}
    
    print("DEBUG (router): state.documents =", state.documents)
    
    messages = list(state.messages)
    
    # Add a message about the current state
    current_state = {
        "documents": list(state.documents.keys()) if state.documents else [],
        "extracted_info": list(state.extracted_info.keys()) if state.extracted_info else [],
        "current_action": state.current_action.action if state.current_action else None,
        "next_actions_remaining": len(state.next_actions) if hasattr(state, "next_actions") else 0
    }
    
    print("Current state:", current_state)
    print("Extracted info keys:", list(state.extracted_info.keys()))
    
    # Determine next action
    if hasattr(state, "next_actions") and state.next_actions:
        # Use the next planned action if available
        next_action = state.next_actions.pop(0)
        state.current_action = next_action
        print(f"Router selected next action: {next_action.action}")
        return {"state": state, "next": next_action.action}
    else:
        # Use the router to determine the next action
        router_prompt = create_router_prompt()
        
        # Format the prompt with the current state and query
        formatted_prompt = router_prompt.format(
            messages=messages,
            query=state.query.query_text,
            current_state=str(current_state)
        )
        
        # Convert the formatted prompt to messages
        formatted_messages = [
            SystemMessage(content=formatted_prompt.split("\n\n")[0]),
            *messages,
            HumanMessage(content=formatted_prompt.split("\n\n")[-1])
        ]
        
        # Invoke the LLM with the formatted messages
        response = tools.llm.invoke(formatted_messages)
        
        next_action = response.content.strip()
        state.current_action = AgentAction(action=next_action)
        print(f"Router selected next action: {next_action}")
        return {"state": state, "next": next_action}