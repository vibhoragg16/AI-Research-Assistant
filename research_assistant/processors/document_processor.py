from typing import List, Optional
from datetime import datetime
import os

from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

from ..models.document import DocumentMetadata, DocumentChunk

class DocumentProcessor:
    def __init__(self):
        print("DEBUG: Initializing DocumentProcessor")
        try:
            print("DEBUG: Creating HuggingFace embeddings")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("DEBUG: Embeddings created successfully")
            
            print("DEBUG: Creating text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            print("DEBUG: Text splitter created successfully")
            
            # Initialize vector store and retriever
            print("DEBUG: Creating InMemoryStore")
            self.docstore = InMemoryStore()
            print("DEBUG: InMemoryStore created successfully")
            
            print("DEBUG: Creating Chroma vector store")
            self.vectorstore = Chroma(
                collection_name="research_papers",
                embedding_function=self.embeddings,
                persist_directory="chroma_db"
            )
            print("DEBUG: Chroma vector store created successfully")
            
            print("DEBUG: Creating MultiVectorRetriever")
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                id_key="chunk_id",
            )
            print("DEBUG: MultiVectorRetriever created successfully")
            
        except Exception as e:
            print("ERROR in DocumentProcessor initialization:", str(e))
            raise
        
    def process_pdf(self, file_path: str) -> str:
        """Process a PDF file and return the document ID"""
        try:
            print("DEBUG: Starting PDF processing")
            # Generate a unique document ID
            document_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print("DEBUG: Generated document ID:", document_id)
            
            # Load and process the document
            print("DEBUG: Loading PDF file")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            print("DEBUG: Loaded", len(pages), "pages")
            
            # Extract document metadata from first page
            title = os.path.basename(file_path).replace('.pdf', '')
            authors = []
            
            # Extract text and metadata
            print("DEBUG: Extracting text from pages")
            all_texts = []
            for i, page in enumerate(pages):
                all_texts.append(page.page_content)
            
            # Split into chunks
            print("DEBUG: Splitting text into chunks")
            chunks = self.text_splitter.split_text("\n".join(all_texts))
            print("DEBUG: Created", len(chunks), "chunks")
            
            # Create document chunks with metadata
            print("DEBUG: Creating document chunks")
            doc_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                doc_chunk = DocumentChunk(
                    text=chunk,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page_num=i // 2  # Rough approximation
                )
                doc_chunks.append(doc_chunk)
                
                # Store in retriever as Document object
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "chunk_id": chunk_id
                    }
                )
                print(f"DEBUG: Storing chunk {i+1} in docstore")
                self.docstore.mset([(chunk_id, doc)])
            
            # Create embeddings and store in vector store
            print("DEBUG: Creating embeddings and storing in vector store")
            self.vectorstore.add_documents(
                documents=[Document(
                    page_content=chunk.text,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id
                    }
                ) for chunk in doc_chunks],
                ids=[chunk.chunk_id for chunk in doc_chunks]  # Add explicit IDs
            )
            print("DEBUG: Successfully stored embeddings")
            
            # Create document metadata
            metadata = DocumentMetadata(
                title=title,
                authors=authors,
                source="local_upload"
            )
            
            # Return document ID for future reference
            print("DEBUG: Successfully processed document:", document_id)
            return document_id
            
        except Exception as e:
            print("ERROR in process_pdf:", str(e))
            raise
    
    def process_arxiv(self, arxiv_id: str) -> str:
        """Process an arXiv paper and return the document ID"""
        try:
            # Generate a unique document ID
            document_id = f"arxiv_{arxiv_id.replace('.', '_')}"
            
            # Load and process the document
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
            docs = loader.load()
            
            if not docs:
                raise Exception(f"Could not find arXiv paper with ID: {arxiv_id}")
                
            doc = docs[0]
            
            # Extract metadata
            metadata_dict = doc.metadata
            metadata = DocumentMetadata(
                title=metadata_dict.get("Title", "Unknown Title"),
                authors=metadata_dict.get("Authors", "").split(", "),
                publication_date=datetime.strptime(metadata_dict.get("Published", "2020-01-01"), "%Y-%m-%d"),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source="arxiv"
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create document chunks with metadata
            doc_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                doc_chunk = DocumentChunk(
                    text=chunk,
                    document_id=document_id,
                    chunk_id=chunk_id
                )
                doc_chunks.append(doc_chunk)
                
                # Store in retriever
                self.docstore.mset([(chunk_id, {"text": chunk, "document_id": document_id, "chunk_id": chunk_id})])
                
            # Create embeddings and store in vector store
            self.vectorstore.add_texts(
                texts=[chunk.text for chunk in doc_chunks],
                metadatas=[{"document_id": chunk.document_id, "chunk_id": chunk.chunk_id} for chunk in doc_chunks],
                ids=[chunk.chunk_id for chunk in doc_chunks]
            )
            
            print("Added document to state:", document_id)
            return document_id
            
        except Exception as e:
            raise Exception(f"Error processing arXiv paper: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str, document_ids: Optional[List[str]] = None, k: int = 5):
        """Retrieve relevant chunks for a query"""
        try:
            print("DEBUG: Starting chunk retrieval")
            print("DEBUG: Query:", query)
            print("DEBUG: Document IDs:", document_ids)
            print("DEBUG: Number of chunks to retrieve:", k)
            
            # First try direct vector store search
            if document_ids:
                print("DEBUG: Filtering by document IDs")
                filter_dict = {"document_id": {"$in": document_ids}}
                docs = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                print("DEBUG: No document IDs provided, retrieving from all documents")
                docs = self.vectorstore.similarity_search(query, k=k)
            
            print("DEBUG: Retrieved chunks:", len(docs))
            
            # Convert retrieved docs to proper Document objects
            processed_docs = []
            for doc in docs:
                if isinstance(doc, dict):
                    # Convert dict to Document object
                    processed_doc = Document(
                        page_content=doc.get("text", ""),
                        metadata={
                            "document_id": doc.get("document_id"),
                            "chunk_id": doc.get("chunk_id")
                        }
                    )
                else:
                    processed_doc = doc
                processed_docs.append(processed_doc)
                print(f"DEBUG: Chunk {len(processed_docs)}:", 
                      processed_doc.metadata.get("document_id"), 
                      processed_doc.page_content[:100] + "...")
            
            return processed_docs
            
        except Exception as e:
            print("ERROR in retrieve_relevant_chunks:", str(e))
            raise

