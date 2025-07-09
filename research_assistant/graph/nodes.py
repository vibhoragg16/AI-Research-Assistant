from ..models.agent import AgentState
from ..tools.agent_tools import AgentTools
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# Add this helper to all node functions
def print_state_id(node_name, state):
    print(f"DEBUG: state id at {node_name}: {id(state)}")


def process_documents(state: AgentState, tools: AgentTools):
    print_state_id('process_documents', state)
    """Process documents mentioned in the query"""
    query = state.query

    # Initialize documents dictionary if not present
    if not hasattr(state, "documents"):
        state.documents = {}

    if query['document_ids']:
        for doc_id in query['document_ids']:
            if doc_id not in state.documents:
                # Simulate document processing
                state.documents[doc_id] = {"processed": True, "id": doc_id}
                state.messages.append({'type': 'ai', 'content': f"Processed document {doc_id}"})
            elif not state.documents[doc_id].get("processed", False):
                # Mark existing document as processed
                state.documents[doc_id]["processed"] = True
                state.messages.append({'type': 'ai', 'content': f"Marked document {doc_id} as processed"})
    else:
        # Extract potential document references from query
        doc_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Identify any document references in the query.
            These could be filenames, arXiv IDs, DOIs, or other document identifiers.
            Format your response as a comma-separated list of identifiers."""),
            HumanMessage(content=query['query_text'])
        ])
        doc_response = tools.llm.invoke(doc_prompt.format_messages(query=query))

        # Parse potential document IDs (simplified)
        potential_docs = [doc.strip() for doc in doc_response.content.split(",") if doc.strip()]

        if potential_docs:
            # Simulate processing these documents
            for doc_id in potential_docs:
                if doc_id not in state.documents:
                    state.documents[doc_id] = {"processed": True, "id": doc_id}
                elif not state.documents[doc_id].get("processed", False):
                    state.documents[doc_id]["processed"] = True

            state.messages.append({'type': 'ai', 'content': f"Processed documents: {', '.join(potential_docs)}"})
            state.query['document_ids'] = potential_docs
        else:
            state.messages.append({'type': 'ai', 'content': "No document references found in the query."})

    # Debug output
    print("DEBUG: query.document_ids =", query['document_ids'])
    print("DEBUG: state.documents =", state.documents)

    # After processing documents, move to retrieve_information
    return {"state": state, "next": "retrieve_information"}

def retrieve_information(state: AgentState, tools: AgentTools):
    print_state_id('retrieve_information', state)
    """Retrieve relevant information from documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available to retrieve information from."})
        return {"state": state, "next": "route"}

    # Get document IDs to search
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Check if we already have retrieved chunks
        if "retrieved_chunks" in state.extracted_info and state.extracted_info["retrieved_chunks"]:
            print("DEBUG: Already have retrieved chunks, moving to generate_summary")
            return {"state": state, "next": "generate_summary"}

        # Retrieve relevant chunks
        chunks = tools.retrieve_document_chunks(query['query_text'], doc_ids, k=5)

        if not chunks:
            state.messages.append({'type': 'ai', 'content': "No relevant information found in the documents."})
            return {"state": state, "next": "route"}

        # Store the retrieved information
        chunk_info = [
            {
                "text": chunk.page_content,
                "doc_id": chunk.metadata.get("document_id"),
                "chunk_id": chunk.metadata.get("chunk_id")
            }
            for chunk in chunks
        ]
        state.extracted_info["retrieved_chunks"] = chunk_info
        state.extracted_info = dict(state.extracted_info)

        # Add message about retrieval
        state.messages.append({'type': 'ai', 'content': f"Retrieved {len(chunks)} relevant chunks from documents: {', '.join(doc_ids)}"})

        # After retrieving information, move to generate_summary
        return {"state": state, "next": "generate_summary"}

    except Exception as e:
        state.error = f"Error retrieving information: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error retrieving information: {str(e)}"})
        return {"state": state, "next": "route"}

def generate_summary(state: AgentState, tools: AgentTools):
    print_state_id('generate_summary', state)
    """Generate summaries for documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available to summarize."})
        return {"state": state, "next": "route"}

    # Get document IDs to summarize
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Generate summaries
        summaries = {}
        for doc_id in doc_ids:
            # Get summary options from query
            summary_type = "general"
            length = "medium"

            if query['options'] and "summary_type" in query['options']:
                summary_type = query['options']["summary_type"]
            if query['options'] and "length" in query['options']:
                length = query['options']["length"]

            summary = tools.summarize_document(doc_id, summary_type, length)
            summaries[doc_id] = {
                "text": summary.summary_text,
                "type": summary_type,
                "length": length
            }

        # Store the summaries
        state.extracted_info["summaries"] = summaries
        state.extracted_info = dict(state.extracted_info)

        # Add message about summaries
        state.messages.append({'type': 'ai', 'content': f"Generated {summary_type} summaries for documents: {', '.join(doc_ids)}"})

    except Exception as e:
        state.error = f"Error generating summaries: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error generating summaries: {str(e)}"})

    return {"state": state, "next": "route"}

def extract_methodology(state: AgentState, tools: AgentTools):
    print_state_id('extract_methodology', state)
    """Extract methodology information from documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available for methodology extraction."})
        return {"state": state, "next": "route"}

    # Get document IDs to process
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Extract methodologies
        methodologies = {}
        for doc_id in doc_ids:
            methodology = tools.extract_methodology(doc_id)
            methodologies[doc_id] = {
                "approach": methodology.approach,
                "datasets": methodology.datasets,
                "algorithms": methodology.algorithms,
                "evaluation_metrics": methodology.evaluation_metrics,
                "limitations": methodology.limitations
            }

        # Store the methodologies
        state.extracted_info["methodologies"] = methodologies
        state.extracted_info = dict(state.extracted_info)

        # Add message about extraction
        state.messages.append({'type': 'ai', 'content': f"Extracted methodology information from documents: {', '.join(doc_ids)}"})

    except Exception as e:
        state.error = f"Error extracting methodologies: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error extracting methodologies: {str(e)}"})

    return {"state": state, "next": "route"}

def extract_claims(state: AgentState, tools: AgentTools):
    print_state_id('extract_claims', state)
    """Extract key claims from documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available for claim extraction."})
        return {"state": state, "next": "route"}

    # Get document IDs to process
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Extract claims
        all_claims = {}
        for doc_id in doc_ids:
            claims = tools.extract_claims(doc_id)
            all_claims[doc_id] = [
                {"claim": claim.claim, "evidence": claim.evidence, "confidence": claim.confidence}
                for claim in claims
            ]

        # Store the claims
        state.extracted_info["claims"] = all_claims
        state.extracted_info = dict(state.extracted_info)

        # Add message about extraction
        state.messages.append({'type': 'ai', 'content': f"Extracted key claims from documents: {', '.join(doc_ids)}"})

    except Exception as e:
        state.error = f"Error extracting claims: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error extracting claims: {str(e)}"})

    return {"state": state, "next": "route"}

def compare_documents(state: AgentState, tools: AgentTools):
    """Compare multiple documents"""
    query = state.query

    if not state.documents or len(state.documents) < 2:
        state.messages.append({'type': 'ai', 'content': "Need at least two documents for comparison."})
        return {"state": state, "next": "route"}

    # Get document IDs to compare
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    if len(doc_ids) < 2:
        state.messages.append({'type': 'ai', 'content': "Need at least two documents for comparison."})
        return {"state": state, "next": "route"}

    try:
        # Compare documents
        comparison = tools.compare_documents(doc_ids)

        # Store the comparison
        state.extracted_info["comparison"] = {
            "similarities": comparison.similarities,
            "differences": comparison.differences,
            "methodology_comparison": comparison.methodology_comparison,
            "result_comparison": comparison.result_comparison
        }
        state.extracted_info = dict(state.extracted_info)

        # Add message about comparison
        state.messages.append({'type': 'ai', 'content': f"Compared documents: {', '.join(doc_ids)}"})

    except Exception as e:
        state.error = f"Error comparing documents: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error comparing documents: {str(e)}"})

    return {"state": state, "next": "route"}

def generate_citation(state: AgentState, tools: AgentTools):
    """Generate citations for documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available for citation generation."})
        return {"state": state, "next": "route"}

    # Get document IDs to cite
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Get citation style from query
        style = "APA"
        if query['options'] and "style" in query['options']:
            style = query['options']["style"]

        # Generate citations
        citations = {}
        for doc_id in doc_ids:
            citation = tools.generate_citation(doc_id, style)
            citations[doc_id] = {
                "text": citation.citation_text,
                "style": citation.style
            }

        # Store the citations
        state.extracted_info["citations"] = citations
        state.extracted_info = dict(state.extracted_info)

        # Add message about citations
        state.messages.append({'type': 'ai', 'content': f"Generated {style} citations for documents: {', '.join(doc_ids)}"})

    except Exception as e:
        state.error = f"Error generating citations: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error generating citations: {str(e)}"})

    return {"state": state, "next": "route"}

def answer_question(state: AgentState, tools: AgentTools):
    """Answer a specific question about documents"""
    query = state.query

    if not state.documents:
        state.messages.append({'type': 'ai', 'content': "No documents available to answer questions from."})
        return {"state": state, "next": "route"}

    # Get document IDs to search
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    try:
        # Answer the question
        answer = tools.answer_question(query['query_text'], doc_ids)

        # Store the answer
        state.extracted_info["answer"] = answer
        state.extracted_info = dict(state.extracted_info)

        # Add message with the answer
        state.messages.append({'type': 'ai', 'content': f"Answer: {answer}"})

    except Exception as e:
        state.error = f"Error answering question: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error answering question: {str(e)}"})

    return {"state": state, "next": "route"}

def generate_literature_review(state: AgentState, tools: AgentTools):
    """Generate a literature review from documents"""
    query = state.query

    if not state.documents or len(state.documents) < 2:
        state.messages.append({'type': 'ai', 'content': "Need at least two documents for a literature review."})
        return {"state": state, "next": "route"}

    # Get document IDs for the review
    doc_ids = query['document_ids'] if query['document_ids'] else list(state.documents.keys())

    if len(doc_ids) < 2:
        state.messages.append({'type': 'ai', 'content': "Need at least two documents for a literature review."})
        return {"state": state, "next": "route"}

    try:
        # Get focus area from query options
        focus = None
        if query['options'] and "focus" in query['options']:
            focus = query['options']["focus"]

        # Generate the literature review
        review = tools.generate_literature_review(doc_ids, focus)

        # Store the review
        state.extracted_info["literature_review"] = review
        state.extracted_info = dict(state.extracted_info)

        # Add message about the review
        state.messages.append({'type': 'ai', 'content': f"Generated literature review for {len(doc_ids)} documents."})

    except Exception as e:
        state.error = f"Error generating literature review: {str(e)}"
        state.messages.append({'type': 'ai', 'content': f"Error generating literature review: {str(e)}"})

    return {"state": state, "next": "route"}

def provide_final_answer(state: AgentState, tools: AgentTools):
    print("DEBUG: extracted_info at provide_final_answer:", state.extracted_info)
    """Provide a final answer based on all the collected information"""
    query = state.query

    # Prepare the context from all extracted information
    context = []
    

    # Add summaries if available
    if "summaries" in state.extracted_info:
        context.append("Document Summaries:")
        for doc_id, summary in state.extracted_info["summaries"].items():
            context.append(f"Document {doc_id} Summary: {summary['text'][:200]}...")
        context.append("")

    # Add methodologies if available
    if "methodologies" in state.extracted_info:
        context.append("Methodology Information:")
        for doc_id, methodology in state.extracted_info["methodologies"].items():
            context.append(f"Document {doc_id} Approach: {methodology['approach']}")
            if methodology['datasets']:
                context.append(f"Datasets: {', '.join(methodology['datasets'])}")
            if methodology['algorithms']:
                context.append(f"Algorithms: {', '.join(methodology['algorithms'])}")
        context.append("")

    # Add claims if available
    if "claims" in state.extracted_info:
        context.append("Key Claims:")
        for doc_id, claims in state.extracted_info["claims"].items():
            for i, claim in enumerate(claims[:3], 1):  # Limit to top 3 claims per doc
                context.append(f"Document {doc_id} Claim {i}: {claim['claim']}")
        context.append("")

    # Add comparison if available
    if "comparison" in state.extracted_info:
        comparison = state.extracted_info["comparison"]
        context.append("Document Comparison:")
        if comparison['similarities']:
            context.append(f"Similarities: {comparison['similarities'][0]}")
        if comparison['differences']:
            context.append(f"Differences: {comparison['differences'][0]}")
        context.append("")

    # Add literature review if available
    if "literature_review" in state.extracted_info:
        context.append("Literature Review:")
        context.append(state.extracted_info["literature_review"][:300] + "...")
        context.append("")

    # Add direct answer if available
    if "answer" in state.extracted_info:
        context.append("Answer to Query:")
        context.append(state.extracted_info["answer"])
        context.append("")

    # Add citations if available
    if "citations" in state.extracted_info:
        context.append("Citations:")
        for doc_id, citation in state.extracted_info["citations"].items():
            context.append(f"{citation['text']}")
        context.append("")

    # Generate the final answer
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an academic research assistant providing final answers to user queries.
        Synthesize all the information gathered to provide a comprehensive, well-structured response.
        Be specific and cite documents when appropriate. 
        Ensure your answer directly addresses the user's original query."""),
        HumanMessage(content=f"""Original Query: {query['query_text']}
        
        Information gathered:
        {chr(10).join(context)}
        
        Based on all this information, provide a complete and coherent response to the original query.
        """)
    ])

    final_answer = tools.llm.invoke(prompt.format_messages(context=context)).content
    print("DEBUG: Final answer from LLM:", final_answer)
    state.final_answer = final_answer
    state.messages.append({'type': 'ai', 'content': final_answer})

    return {"state": state, "next": "END"}
    