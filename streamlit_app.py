import streamlit as st
import asyncio
import os
import sys
from threading import Thread

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the simplified app
try:
    from simplified_app import ResearchAssistant, QueryType, fix_asyncio
    
    # Apply asyncio fix
    fix_asyncio()
    
    # Initialize app
    st.title("AI Research Assistant - Simple Mode")
    st.info("Running in simplified mode with minimal dependencies")
    
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
    
    # Initialize the research assistant
    @st.cache_resource
    def get_assistant():
        return ResearchAssistant()
    
    assistant = get_assistant()
    
    # Track processed documents
    if "document_ids" not in st.session_state:
        st.session_state.document_ids = []
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the PDF
            try:
                doc_id = assistant.doc_processor.process_pdf(temp_file_path)
                if doc_id not in st.session_state.document_ids:
                    st.session_state.document_ids.append(doc_id)
                    st.success(f"Successfully processed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    # ArXiv ID input
    arxiv_id = st.text_input("Or enter arXiv ID (e.g., 2303.08774)")
    if arxiv_id and st.button("Process arXiv paper"):
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
        st.subheader("Query Options")
        
        query_type = st.selectbox(
            "Select query type",
            options=[
                QueryType.SUMMARIZE,
                QueryType.EXTRACT,
                QueryType.COMPARE,
                QueryType.QUESTION
            ],
            format_func=lambda x: x.capitalize()
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
                with st.spinner("Processing query..."):
                    try:
                        # Use the run_async helper function
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
                        import traceback
                        st.code(traceback.format_exc(), language="python")
    
    # Footer
    st.markdown("---")
    st.markdown("AI Research Assistant - Simplified Mode")

except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.warning("Please make sure simplified_app.py is in the same directory as this file.")
    
    # Show debug info
    st.subheader("Debugging Information")
    st.write("Current directory:", os.getcwd())
    st.write("Python path:", sys.path)
