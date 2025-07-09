from typing import List, Dict, Any, Optional
import os
import asyncio

from .models.query import QueryType, AgentQuery
from .models.agent import AgentState
from .processors.document_processor import DocumentProcessor
from .tools.agent_tools import AgentTools
from .graph.workflow import setup_graph
from langchain_core.messages import SystemMessage, HumanMessage

class ResearchAssistant:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.tools = AgentTools(self.doc_processor)
        # self.graph = setup_graph(self.tools)
    
    def process_paper(self, file_path_or_id):
        """Process a paper from file or arXiv ID"""
        if file_path_or_id.endswith('.pdf'):
            # Local PDF file
            return self.doc_processor.process_pdf(file_path_or_id)
        elif '.' in file_path_or_id and not os.path.exists(file_path_or_id):
            # Likely an arXiv ID
            return self.doc_processor.process_arxiv(file_path_or_id)
        else:
            raise ValueError("Unsupported document format or ID")
    
    async def run(self, query_text, query_type, document_ids=None, options=None):
        """Run the research assistant on a query"""
        # Create the query object
        query = {
            'query_type': query_type,
            'query_text': query_text,
            'document_ids': document_ids,
            'options': options
        }
        
        # Create initial state
        state = AgentState(
            query=query,
            messages=[{'type': 'human', 'content': query_text}]
        )
        
        # Direct function call workflow
        from .graph import nodes
        state = nodes.process_documents(state, self.tools)["state"]
        state = nodes.retrieve_information(state, self.tools)["state"]
        state = nodes.generate_summary(state, self.tools)["state"]
        state = nodes.provide_final_answer(state, self.tools)["state"]
        return state


# Example usage
async def example():
    # Initialize the research assistant
    assistant = ResearchAssistant()
    
    # Process papers (in real usage, you'd have actual PDFs or arXiv IDs)
    doc_id1 = "sample_doc_1"  # Simulated document ID
    doc_id2 = "sample_doc_2"  # Simulated document ID
    
    # Run a query to summarize documents
    result = await assistant.run(
        query_text="Summarize these papers and compare their methodologies",
        query_type=QueryType.SUMMARIZE,
        document_ids=[doc_id1, doc_id2],
        options={"summary_type": "general", "length": "medium"}
    )
    
    # Print the final answer
    print(result.final_answer)


# Entry point for running the example
if __name__ == "__main__":
    import asyncio
    asyncio.run(example())