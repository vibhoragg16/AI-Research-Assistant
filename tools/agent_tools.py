from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Optional


from ..processors.document_processor import DocumentProcessor
from ..models.document import DocumentSummary
from ..models.research import MethodologyInfo, ResearchClaim, ComparisonResult, Citation

class AgentTools:
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor
        self.llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
    
    def retrieve_document_chunks(self, query: str, document_ids: Optional[List[str]] = None, k: int = 5):
        """Retrieve relevant document chunks for a query"""
        return self.doc_processor.retrieve_relevant_chunks(query, document_ids, k)
    
    def summarize_document(self, document_id: str, summary_type: str = "general", length: str = "medium") -> DocumentSummary:
        """Generate a summary of the document"""
        # Retrieve chunks from the document
        chunks = self.retrieve_document_chunks(
            f"Create a {summary_type} summary", 
            document_ids=[document_id], 
            k=10
        )
        
        # Combine chunks into context
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Create the summary prompt based on type and length
        length_guidance = {
            "short": "1-2 paragraphs",
            "medium": "3-5 paragraphs",
            "long": "comprehensive, 6+ paragraphs"
        }
        
        type_guidance = {
            "general": "overall summary focusing on the main contributions and findings",
            "methods": "summary focusing on the methodology, experimental setup, and technical approaches",
            "results": "summary focusing on the results, evaluations, and outcomes of the research",
            "background": "summary focusing on the background, related work, and context of the research"
        }
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a specialized academic summarization agent. 
            Create a {length_guidance[length]} {type_guidance.get(summary_type, type_guidance['general'])} 
            of the academic content provided. 
            Focus on accuracy and capturing the essential information."""),
            HumanMessage(content=f"Here is the content to summarize:\n\n{context}")
        ])
        
        # Generate summary
        summary_text = self.llm.invoke(prompt.format_messages()).content
        
        # Create summary object
        summary = DocumentSummary(
            document_id=document_id,
            summary_text=summary_text,
            summary_type=summary_type,
            length=length
        )
        
        return summary
    
    def extract_methodology(self, document_id: str) -> MethodologyInfo:
        """Extract methodology information from a document"""
        # Retrieve chunks likely to contain methodology information
        chunks = self.retrieve_document_chunks(
            "methodology experimental setup methods algorithm approach", 
            document_ids=[document_id],
            k=8
        )
        
        # Combine chunks into context
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Create extraction prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a specialized methodology extraction agent for academic papers.
            Extract the following information from the provided text:
            1. The overall research approach or methodology
            2. Datasets used in the research
            3. Algorithms or models implemented
            4. Evaluation metrics used
            5. Limitations mentioned about the methodology
            
            Format your response as a structured list. If certain information is not present, indicate this."""),
            HumanMessage(content=f"Here is the content to analyze:\n\n{context}")
        ])
        
        # Generate extraction
        extraction_text = self.llm.invoke(prompt.format_messages()).content
        
        # Use the LLM to convert the extraction text to a structured format
        structure_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Parse the following methodology extraction into a structured format.
            Extract exactly these fields:
            - approach: The overall research approach
            - datasets: List of datasets used
            - algorithms: List of algorithms or models used
            - evaluation_metrics: List of evaluation metrics
            - limitations: List of methodology limitations
            
            Format as JSON."""),
            HumanMessage(content=extraction_text)
        ])
        
        structure_text = self.llm.invoke(structure_prompt).content
        
        # Parse the JSON (in a real implementation, we'd handle this more robustly)
        import json
        try:
            structure = json.loads(structure_text)
            
            # Create methodology object
            methodology = MethodologyInfo(
                approach=structure.get("approach", "Not specified"),
                datasets=structure.get("datasets", []),
                algorithms=structure.get("algorithms", []),
                evaluation_metrics=structure.get("evaluation_metrics", []),
                limitations=structure.get("limitations", []),
                document_id=document_id
            )
            
            return methodology
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return MethodologyInfo(
                approach="Extraction failed - could not parse response",
                document_id=document_id
            )
    
    def extract_claims(self, document_id: str) -> List[ResearchClaim]:
        """Extract key claims from a document"""
        # Retrieve chunks from the document
        chunks = self.retrieve_document_chunks(
            "key findings results conclusions claims contributions", 
            document_ids=[document_id],
            k=8
        )
        
        # Combine chunks into context
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Create extraction prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a specialized claim extraction agent for academic papers.
            Extract the top 3-5 key claims or findings from the provided text.
            For each claim, identify:
            1. The claim statement
            2. Supporting evidence from the text
            3. A confidence score (0.0-1.0) based on the strength of evidence
            
            Format your response as a list of claims with these components."""),
            HumanMessage(content=f"Here is the content to analyze:\n\n{context}")
        ])
        
        # Generate extraction
        extraction_text = self.llm.invoke(prompt.format_messages()).content
        
        # Use the LLM to convert the extraction text to a structured format
        structure_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Parse the following claim extraction into a structured format.
            Format each claim as an object with these fields:
            - claim: The claim statement
            - evidence: Supporting evidence text
            - confidence: A number from 0.0 to 1.0
            
            Format as a JSON array of claim objects."""),
            HumanMessage(content=extraction_text)
        ])
        
        structure_text = self.llm.invoke(structure_prompt).content
        
        # Parse the JSON (in a real implementation, we'd handle this more robustly)
        import json
        try:
            claims_data = json.loads(structure_text)
            
            # Create claim objects
            claims = []
            for claim_data in claims_data:
                claim = ResearchClaim(
                    claim=claim_data.get("claim", ""),
                    evidence=claim_data.get("evidence", ""),
                    confidence=float(claim_data.get("confidence", 0.5)),
                    document_id=document_id
                )
                claims.append(claim)
            
            return claims
            
        except (json.JSONDecodeError, ValueError):
            # Fallback if parsing fails
            return [ResearchClaim(
                claim="Extraction failed - could not parse response",
                evidence="",
                confidence=0.0,
                document_id=document_id
            )]
    
    def compare_documents(self, document_ids: List[str]) -> ComparisonResult:
        """Compare multiple documents"""
        if len(document_ids) < 2:
            raise ValueError("Need at least two documents to compare")
            
        # Get summaries for each document
        summaries = []
        for doc_id in document_ids:
            summary = self.summarize_document(doc_id, "general", "medium")
            summaries.append((doc_id, summary.summary_text))
            
        # Extract methodologies
        methodologies = []
        for doc_id in document_ids:
            methodology = self.extract_methodology(doc_id)
            methodologies.append((doc_id, methodology))
            
        # Create comparison context
        comparison_context = "Documents to compare:\n\n"
        for i, (doc_id, summary) in enumerate(summaries, 1):
            comparison_context += f"Document {i} ({doc_id}):\n{summary}\n\n"
            
        # Create comparison prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a specialized comparison agent for academic papers.
            Compare the provided documents and identify:
            1. Key similarities in approach, methods, or findings
            2. Key differences in approach, methods, or findings
            3. A comparison of methodologies used
            4. A comparison of results and conclusions
            
            Be specific and reference the document identifiers in your comparison."""),
            HumanMessage(content=comparison_context)
        ])
        
        # Generate comparison
        comparison_text = self.llm.invoke(prompt.format_messages()).content
        
        # Use the LLM to convert to a structured format
        structure_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Parse the following comparison into a structured format.
            Extract these components:
            - similarities: List of key similarities
            - differences: List of key differences
            - methodology_comparison: Comparison of methodologies
            - result_comparison: Comparison of results
            
            Format as JSON."""),
            HumanMessage(content=comparison_text)
        ])
        
        structure_text = self.llm.invoke(structure_prompt).content
        
        # Parse the JSON
        import json
        try:
            structure = json.loads(structure_text)
            
            # Create comparison result
            comparison = ComparisonResult(
                similarities=structure.get("similarities", []),
                differences=structure.get("differences", []),
                methodology_comparison=structure.get("methodology_comparison", ""),
                result_comparison=structure.get("result_comparison", ""),
                document_ids=document_ids
            )
            
            return comparison
            
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return ComparisonResult(
                similarities=["Comparison failed - could not parse response"],
                differences=["Comparison failed - could not parse response"],
                document_ids=document_ids
            )
    
    def generate_citation(self, document_id: str, style: str = "APA") -> Citation:
        """Generate a citation for a document"""
        # We'd normally look up document metadata from a database
        # For this example, we'll retrieve some text from the document to infer metadata
        chunks = self.retrieve_document_chunks(
            "title authors publication", 
            document_ids=[document_id],
            k=2
        )
        
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Create citation prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a citation generation agent.
            Based on the provided text from an academic document, generate a {style} style citation.
            If information is missing, make reasonable assumptions but indicate uncertainty."""),
            HumanMessage(content=f"Here is text from the document:\n\n{context}")
        ])
        
        # Generate citation
        citation_text = self.llm.invoke(prompt.format_messages()).content
        
        # Create citation object
        citation = Citation(
            document_id=document_id,
            citation_text=citation_text,
            style=style
        )
        
        return citation
    
    def answer_question(self, question: str, document_ids: Optional[List[str]] = None) -> str:
        """Answer a question based on document content"""
        # Retrieve relevant chunks
        chunks = self.retrieve_document_chunks(question, document_ids, k=8)
        
        if not chunks:
            return "I couldn't find relevant information to answer this question."
            
        # Combine chunks into context
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        
        # Get document reference info for citations
        doc_refs = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id")
            if doc_id and doc_id not in doc_refs:
                # In a real implementation, we'd get actual metadata
                doc_refs[doc_id] = f"Document {doc_id}"
        
        # Create answering prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a specialized research question answering agent.
            Answer the question based solely on the provided context.
            Be specific and cite the relevant documents using [Document ID] notation.
            If the context doesn't contain enough information to answer, say so clearly."""),
            HumanMessage(content=f"""Question: {question}
            
            Context information:
            {context}
            
            Document references:
            {', '.join([f'{id}: {ref}' for id, ref in doc_refs.items()])}""")
        ])
        
        # Generate answer
        answer = self.llm.invoke(prompt.format_messages()).content
        
        return answer
    
    def generate_literature_review(self, document_ids: List[str], focus: Optional[str] = None) -> str:
        """Generate a literature review from multiple documents"""
        # Get summaries for each document
        summaries = []
        for doc_id in document_ids:
            summary = self.summarize_document(doc_id, "general", "medium")
            summaries.append((doc_id, summary.summary_text))
            
        # Extract methodologies and claims
        methodologies = []
        all_claims = []
        for doc_id in document_ids:
            methodology = self.extract_methodology(doc_id)
            claims = self.extract_claims(doc_id)
            methodologies.append((doc_id, methodology))
            all_claims.extend(claims)
            
        # Create literature review context
        review_context = "Documents for literature review:\n\n"
        for i, (doc_id, summary) in enumerate(summaries, 1):
            review_context += f"Document {i} ({doc_id}):\nSummary: {summary}\n\n"
            
            # Add methodology info
            for meth_id, meth in methodologies:
                if meth_id == doc_id:
                    review_context += f"Methodology: {meth.approach}\n"
                    if meth.datasets:
                        review_context += f"Datasets: {', '.join(meth.datasets)}\n"
                    if meth.algorithms:
                        review_context += f"Algorithms: {', '.join(meth.algorithms)}\n\n"
            
            # Add key claims
            review_context += "Key claims:\n"
            for claim in all_claims:
                if claim.document_id == doc_id:
                    review_context += f"- {claim.claim}\n"
            review_context += "\n"
            
        # Create review prompt
        focus_text = f" with a focus on {focus}" if focus else ""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a specialized literature review agent.
            Generate a comprehensive literature review of the provided documents{focus_text}.
            Include:
            1. An overview of the field and research questions
            2. Analysis of methodologies used across papers
            3. Synthesis of key findings and claims
            4. Identification of research gaps or contradictions
            5. Suggestions for future research directions
            
            Reference specific documents using [Document ID] notation."""),
            HumanMessage(content=review_context)
        ])
        
        # Generate literature review
        review = self.llm.invoke(prompt.format_messages()).content
        
        return review

