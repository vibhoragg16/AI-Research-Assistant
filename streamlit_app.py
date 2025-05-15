import streamlit as st
import os
import asyncio
import nest_asyncio
from research_assistant.app import ResearchAssistant
from research_assistant.models.query import QueryType
from research_assistant.config import init_environment

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

init_environment()

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

# Initialize the assistant
assistant = ResearchAssistant()

# Sidebar for document input
st.sidebar.header("Document Input")
doc_source = st.sidebar.radio("Choose document source:", ("Upload PDF", "arXiv ID"))

uploaded_file = None
arxiv_id = None

document_ids = []

if doc_source == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        # Save uploaded file to a temp location
        temp_path = os.path.join("temp_" + uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            doc_id = assistant.process_paper(temp_path)
            document_ids.append(doc_id)
            st.sidebar.success(f"PDF processed. Document ID: {doc_id}")
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {e}")
else:
    arxiv_id = st.sidebar.text_input("Enter arXiv ID (e.g., 1706.03762)")
    if arxiv_id:
        try:
            doc_id = assistant.process_paper(arxiv_id)
            document_ids.append(doc_id)
            st.sidebar.success(f"arXiv paper processed. Document ID: {doc_id}")
        except Exception as e:
            st.sidebar.error(f"Error processing arXiv paper: {e}")

# Main panel for query
st.header("Ask a Research Question")
query_type = st.selectbox(
    "Select query type:",
    [qtype.value for qtype in QueryType]
)

query_text = st.text_area("Enter your query/question:")

options = {}
if query_type == QueryType.SUMMARIZE.value:
    options["summary_type"] = st.selectbox("Summary type:", ["general", "methods", "results", "background"])
    options["length"] = st.selectbox("Summary length:", ["short", "medium", "long"])

run_button = st.button("Run Research Assistant")

result_placeholder = st.empty()

def run_query():
    """Run the query synchronously"""
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the query
        result = loop.run_until_complete(assistant.run(
            query_text=query_text,
            query_type=query_type,
            document_ids=document_ids if document_ids else None,
            options=options if options else None
        ))
        
        # Close the loop
        loop.close()
        
        return result
    except Exception as e:
        st.error(f"Error running query: {str(e)}")
        return None

if run_button:
    if not query_text:
        st.warning("Please enter a query/question.")
    elif not document_ids:
        st.warning("Please upload a PDF or enter an arXiv ID.")
    else:
        with st.spinner("Running research assistant..."):
            try:
                result = run_query()
                if result and hasattr(result, "final_answer"):
                    result_placeholder.success(result.final_answer)
                else:
                    result_placeholder.info(str(result))
            except Exception as e:
                result_placeholder.error(f"Error: {e}") 