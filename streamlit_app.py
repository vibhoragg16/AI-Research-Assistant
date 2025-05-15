import streamlit as st
import asyncio
import nest_asyncio
import os
import tempfile
import sys
from threading import Thread
from typing import List, Dict, Any, Optional

# Apply nest_asyncio at the start
nest_asyncio.apply()

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Function to run async tasks in Streamlit
def run_async(coro):
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
    
    thread = Thread(target=run_in_thread)
    thread.start()
    thread.join()
    
    if exception:
        raise exception
    
    return result

# Import research assistant modules
try:
    from research_assistant.app import ResearchAssistant
    from research_assistant.models.query import QueryType
    from research_assistant.config import init_environment
    
    # Initialize environment variables
    init_environment()
    
    # Set page config
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize the research assistant
    @st.cache_resource
    def get_assistant():
        try:
            assistant = ResearchAssistant()
            return assistant
        except Exception as e:
            st.error(f"Error initializing Research Assistant: {str(e)}")
            return None
    
    # App title and description
    st.title("📚 AI Research Assistant")
    st.markdown("""
    Upload academic papers (PDF) or provide arXiv IDs to analyze research content.
    This tool can summarize papers, extract key information, compare multiple papers, or answer specific questions.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool uses AI to analyze academic papers and provide insights.
        
        ### Features:
        - Summarize research papers
        - Extract key information
        - Compare multiple papers
        - Answer questions about paper content
        
        ### How to use:
        1. Upload PDFs or enter arXiv IDs
        2. Select query type
        3. Configure options
        4. Enter your query
        5. Get AI-generated insights
        """)
        
        st.divider()
        st.markdown("Built with LangChain and Groq LLMs")
    
    # Main content
    tab1, tab2 = st.tabs(["Document Processing", "Analysis"])
    
    with tab1:
        # Track processed documents
        if "document_ids" not in st.session_state:
            st.session_state.document_ids = []
            st.session_state.document_names = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Papers")
            uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)
            
            if uploaded_files:
                assistant = get_assistant()
                if assistant:
                    for uploaded_file in uploaded_files:
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_file_path = tmp_file.name
                        
                        # Process the PDF
                        try:
                            with st.spinner(f"Processing {uploaded_file.name}..."):
                                doc_id = assistant.doc_processor.process_pdf(temp_file_path)
                                if doc_id not in st.session_state.document_ids:
                                    st.session_state.document_ids.append(doc_id)
                                    st.session_state.document_names[doc_id] = uploaded_file.name
                                    st.success(f"Successfully processed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
        
        with col2:
            st.subheader("ArXiv Papers")
            arxiv_id = st.text_input("Enter arXiv ID (e.g., 2303.08774)")
            
            if arxiv_id and st.button("Process arXiv paper"):
                assistant = get_assistant()
                if assistant:
                    try:
                        with st.spinner(f"Processing arXiv:{arxiv_id}..."):
                            doc_id = assistant.doc_processor.process_arxiv(arxiv_id)
                            if doc_id not in st.session_state.document_ids:
                                st.session_state.document_ids.append(doc_id)
                                st.session_state.document_names[doc_id] = f"arXiv:{arxiv_id}"
                                st.success(f"Successfully processed paper with arXiv ID: {arxiv_id}")
                    except Exception as e:
                        st.error(f"Error processing arXiv paper: {str(e)}")
        
        # Show processed documents
        if st.session_state.document_ids:
            st.subheader("Processed Documents")
            for i, doc_id in enumerate(st.session_state.document_ids):
                doc_name = st.session_state.document_names.get(doc_id, doc_id)
                st.write(f"{i+1}. {doc_name} (ID: {doc_id})")
                
            if st.button("Clear all documents"):
                st.session_state.document_ids = []
                st.session_state.document_names = {}
                st.experimental_rerun()
    
    with tab2:
        if not st.session_state.document_ids:
            st.info("Please process at least one document in the 'Document Processing' tab first.")
        else:
            st.subheader("Analysis Options")
            
            query_type = st.selectbox(
                "Select analysis type",
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
                col1, col2 = st.columns(2)
                with col1:
                    options["summary_type"] = st.selectbox(
                        "Summary focus",
                        options=["general", "methodology", "results", "conclusions"]
                    )
                with col2:
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
                if len(st.session_state.document_ids) < 2:
                    st.warning("Please process at least two documents for comparison.")
                
                options["comparison_focus"] = st.multiselect(
                    "Comparison focus",
                    options=["methodology", "results", "datasets", "conclusions"],
                    default=["methodology", "results"]
                )
            
            # Query input
            query_text = ""
            if query_type == QueryType.QUESTION:
                query_text = st.text_area("Enter your specific question about the papers", height=100)
            else:
                default_prompts = {
                    QueryType.SUMMARIZE: f"Provide a {options.get('length', 'medium')} {options.get('summary_type', 'general')} summary of the paper(s)",
                    QueryType.EXTRACT: f"Extract the {options.get('extract_type', 'key findings')} from the paper(s)",
                    QueryType.COMPARE: f"Compare the papers focusing on {', '.join(options.get('comparison_focus', ['methodology', 'results']))}"
                }
                query_text = st.text_area("Customize your request (optional)", 
                                          value=default_prompts.get(query_type, ""),
                                          height=100)
            
            # Run button
            if st.button("Run Analysis", type="primary"):
                if query_type == QueryType.QUESTION and not query_text:
                    st.warning("Please enter a question")
                else:
                    assistant = get_assistant()
                    if assistant:
                        with st.spinner("Processing analysis..."):
                            try:
                                # Use run_async to handle asyncio
                                result = run_async(assistant.run(
                                    query_text=query_text,
                                    query_type=query_type,
                                    document_ids=st.session_state.document_ids,
                                    options=options
                                ))
                                
                                # Display the results
                                st.subheader("Results")
                                if hasattr(result, 'final_answer') and result.final_answer:
                                    st.markdown(result.final_answer)
                                else:
                                    st.write(result)
                                
                            except Exception as e:
                                st.error(f"Error processing analysis: {str(e)}")
                                with st.expander("View error details"):
                                    import traceback
                                    st.code(traceback.format_exc(), language="python")

except ImportError as e:
    st.error(f"Import error: {str(e)}")
    
    # Provide guidance
    with st.expander("Troubleshooting Information"):
        st.markdown("""
        ### Possible issues:
        
        1. **Package dependencies missing**
           - Make sure you've installed all required packages:
           ```
           pip install streamlit langchain langchain-core nest-asyncio python-dotenv
           ```
           
        2. **Project structure issue**
           - Ensure your project structure matches what's expected:
           ```
           project_root/
           ├── research_assistant/
           │   ├── __init__.py
           │   ├── app.py
           │   ├── config.py
           │   ├── models/
           │   ├── processors/
           │   ├── tools/
           │   └── graph/
           └── streamlit_app.py (this file)
           ```
           
        3. **Python path issue**
           - Make sure your project root is in the Python path
        
        4. **Asyncio event loop issue**
           - The error about 'no running event loop' happens when trying to use asyncio in Streamlit
           - This app includes a fix for that using nest_asyncio and threaded execution
        """)
        
        st.subheader("System Information")
        st.write("Current directory:", os.getcwd())
        st.write("Python path:", sys.path)
        try:
            import pkg_resources
            st.write("Installed packages:")
            packages = sorted([f"{p.key} {p.version}" for p in pkg_resources.working_set])
            st.code("\n".join(packages[:20] + ["..." if len(packages) > 20 else ""]))
        except:
            st.write("Could not list installed packages")
