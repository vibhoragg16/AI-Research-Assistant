import streamlit as st
import os
import asyncio
import nest_asyncio
from research_assistant.app import ResearchAssistant
from research_assistant.models.query import QueryType
from research_assistant.config import init_environment
import time
import tempfile
from langchain_core.messages import HumanMessage
# SQLite compatibility fix for Streamlit Cloud
import sys
import subprocess

# Install pysqlite3-binary if not present

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize environment
init_environment()

# Fix for Streamlit watcher issues
import streamlit.watcher.local_sources_watcher as watcher

def safe_get_module_paths(module):
    try:
        return watcher.extract_paths(module)
    except Exception:
        return []

watcher.get_module_paths = safe_get_module_paths

# Set up the Streamlit interface
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

# Initialize session state for tracking documents
if 'document_ids' not in st.session_state:
    st.session_state.document_ids = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = ResearchAssistant()

# Sidebar for document input
st.sidebar.header("Document Input")
doc_source = st.sidebar.radio("Choose document source:", ("Upload PDF", "arXiv ID"))

# Document upload/processing section
if doc_source == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        # Create a temp file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getvalue())
        
        # Process button with clear visual feedback
        process_pdf = st.sidebar.button("Process PDF")
        if process_pdf:
            try:
                with st.sidebar.status("Processing PDF..."):
                    doc_id = st.session_state.assistant.process_paper(temp_path)
                    if doc_id not in st.session_state.document_ids:
                        st.session_state.document_ids.append(doc_id)
                st.sidebar.success(f"PDF processed successfully! Document ID: {doc_id}")
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {str(e)}")
                st.sidebar.info("Check console for detailed error messages.")
else:
    arxiv_id = st.sidebar.text_input("Enter arXiv ID (e.g., 1706.03762)")
    process_arxiv = st.sidebar.button("Process arXiv Paper")
    
    if process_arxiv and arxiv_id:
        try:
            with st.sidebar.status("Processing arXiv paper..."):
                doc_id = st.session_state.assistant.process_paper(arxiv_id)
                if doc_id not in st.session_state.document_ids:
                    st.session_state.document_ids.append(doc_id)
            st.sidebar.success(f"arXiv paper processed! Document ID: {doc_id}")
        except Exception as e:
            st.sidebar.error(f"Error processing arXiv paper: {str(e)}")
            st.sidebar.info("Check console for detailed error messages.")

# Display currently loaded documents
if st.session_state.document_ids:
    st.sidebar.subheader("Loaded Documents")
    for i, doc_id in enumerate(st.session_state.document_ids):
        st.sidebar.text(f"{i+1}. {doc_id}")
    
    if st.sidebar.button("Clear All Documents"):
        st.session_state.document_ids = []
        st.sidebar.success("All documents cleared!")

# Main panel for query
st.header("Ask a Research Question")

# Only allow queries if documents are loaded
if not st.session_state.document_ids:
    st.warning("Please load at least one document before submitting a query.")

# Query interface
query_type = st.selectbox(
    "Select query type:",
    [qtype.value for qtype in QueryType]
)

query_text = st.text_area("Enter your query/question:")

# Dynamic options based on query type
options = {}
if query_type == QueryType.SUMMARIZE.value:
    col1, col2 = st.columns(2)
    with col1:
        options["summary_type"] = st.selectbox("Summary type:", ["general", "methods", "results", "background"])
    with col2:
        options["length"] = st.selectbox("Summary length:", ["short", "medium", "long"])

# Result display area with proper placeholder
result_container = st.container()
result_placeholder = st.empty()

# Execute query
run_button = st.button("Run Research Assistant", disabled=not st.session_state.document_ids)

async def run_async_query():
    """Execute the query asynchronously and return the result"""
    try:
        # Create query and send to assistant
        result = await st.session_state.assistant.run(
            query_text=query_text,
            query_type=query_type,
            document_ids=st.session_state.document_ids,
            options=options
        )
        return result
    except Exception as e:
        import traceback
        st.error(f"Error running query: {str(e)}")
        st.code(traceback.format_exc())
        return None

if run_button and query_text and st.session_state.document_ids:
    with st.status("Processing your query...", expanded=True) as status:
        try:
            # Create a progress indicator
            progress_text = "Running research assistant..."
            progress_bar = st.progress(0)
            
            # Execute the query
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Update progress to show activity
            for percent_complete in range(0, 101, 10):
                time.sleep(0.1)  # Simulate processing time
                progress_bar.progress(percent_complete)
            
            # Actually run the query
            result = loop.run_until_complete(run_async_query())
            loop.close()
            
            # Update status
            progress_bar.progress(100)
            status.update(label="Query complete!", state="complete", expanded=False)
            
            # Display the result
            if result:
                if hasattr(result, "final_answer") and result.final_answer:
                    # Show the final answer if available
                    with result_container:
                        st.subheader("Research Assistant Response")
                        st.markdown(result.final_answer)
                else:
                    # Show a human-readable message when no final answer
                    with result_container:
                        st.info("The assistant processed your query but didn't generate a final answer. This could be due to insufficient information in the documents.")
            else:
                with result_container:
                    st.error("Failed to generate a response. Please check the error messages and try again.")
        
        except Exception as e:
            import traceback
            st.error(f"Unexpected error: {str(e)}")
            st.code(traceback.format_exc())
            status.update(label="Error occurred", state="error")

# Add a debug section that can be expanded if needed
with st.expander("Debug Information"):
    st.write("Document IDs:", st.session_state.document_ids)
    st.write("Query Type:", query_type)
    st.write("Query Options:", options)
    
    if st.button("Print Session State"):
        filtered_state = {k: v for k, v in st.session_state.items() 
                          if k not in ['assistant']}  # Filter out complex objects
        st.write(filtered_state)
