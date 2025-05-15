import streamlit as st
import asyncio
import nest_asyncio
from threading import Thread
import sys
import os
from typing import List, Optional

# Add the project root to Python path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your research assistant module
from research_assistant.app import ResearchAssistant
from research_assistant.models.query import QueryType

# Apply nest_asyncio to make asyncio work in Streamlit
nest_asyncio.apply()

# Initialize the research assistant
@st.cache_resource
def get_assistant():
    return ResearchAssistant()

# Function to run async tasks in a way compatible with Streamlit
def run_async(coro):
    # This creates a new event loop in a separate thread and runs the coroutine there
    result = None
    exception = None
    
    def run_in_thread():
        nonlocal result, exception
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            loop.close()
    
    # Create and start the thread
    thread = Thread(target=run_in_thread)
    thread.start()
    thread.join()
    
    # If there was an exception, raise it
    if exception:
        raise exception
    
    return result

# Streamlit UI
st.title("AI Research Assistant")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)

# Track processed documents
if "document_ids" not in st.session_state:
    st.session_state.document_ids = []

# Process uploaded files
if uploaded_files:
    assistant = get_assistant()
    
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF and store the document ID
        try:
            doc_id = assistant.doc_processor.process_pdf(temp_file_path)
            if doc_id not in st.session_state.document_ids:
                st.session_state.document_ids.append(doc_id)
                st.success(f"Successfully processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

# ArXiv ID input
arxiv_id = st.text_input("Or enter arXiv ID (e.g., 2303.08774)")
if arxiv_id and st.button("Process arXiv paper"):
    assistant = get_assistant()
    try:
        doc_id = assistant.doc_processor.process_arxiv(arxiv_id)
        if doc_id not in st.session_state.document_ids:
            st.session_state.document_ids.append(doc_id)
            st.success(f"Successfully processed paper with arXiv ID: {arxiv_id}")
    except Exception as e:
        st.error(f"Error processing arXiv paper: {str(e)}")

# Show processed documents
if st.session_state.document_ids:
    st.subheader("Processed Documents")
    for i, doc_id in enumerate(st.session_state.document_ids):
        st.write(f"{i+1}. Document ID: {doc_id}")

# Query options
if st.session_state.document_ids:
    st.subheader("Query Options")
    
    query_type = st.selectbox(
        "Select query type",
        options=[
            QueryType.SUMMARIZE,
            QueryType.EXTRACT,
            QueryType.COMPARE,
            QueryType.QUESTION
        ],
        format_func=lambda x: x.value.capitalize()
    )
    
    # Different options based on query type
    options = {}
    
    if query_type == QueryType.SUMMARIZE:
        options["summary_type"] = st.selectbox(
            "Summary type",
            options=["general", "methodology", "results", "conclusions"]
        )
        options["length"] = st.select_slider(
            "Summary length",
            options=["very short", "short", "medium", "long", "very long"],
            value="medium"
        )
    
    elif query_type == QueryType.EXTRACT:
        options["extract_type"] = st.selectbox(
            "Extract type",
            options=["key findings", "methods", "datasets", "metrics", "limitations"]
        )
    
    elif query_type == QueryType.COMPARE:
        options["comparison_focus"] = st.multiselect(
            "Comparison focus",
            options=["methodology", "results", "datasets", "conclusions"],
            default=["methodology", "results"]
        )
    
    query_text = st.text_area("Enter your query", height=100)
    
    if st.button("Run Query"):
        if not query_text:
            st.warning("Please enter a query")
        else:
            assistant = get_assistant()
            with st.spinner("Processing query..."):
                try:
                    # Use the run_async helper function to handle asyncio
                    result = run_async(assistant.run(
                        query_text=query_text,
                        query_type=query_type,
                        document_ids=st.session_state.document_ids,
                        options=options
                    ))
                    
                    # Display the results
                    st.subheader("Results")
                    if hasattr(result, 'final_answer'):
                        st.markdown(result.final_answer)
                    else:
                        st.write(result)
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")

# Add a footer
st.markdown("---")
st.markdown("AI Research Assistant | Powered by LangChain and Groq")
