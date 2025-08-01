import os
import logging
from typing import Dict
from datetime import datetime

from langchain_community.retrievers import ArxivRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivLearningAgent:
    """
    Simple LangChain agent for learning from Arxiv papers and answering questions.
    
    Three main functions:
    1. Learn about topics by downloading Arxiv papers
    2. Answer questions using RAG on stored knowledge
    3. Explore what topics the system knows about
    """
    
    def __init__(self, persist_directory: str = "./knowledge_base"):
        """Initialize the agent with necessary components."""
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize or load existing vector store
        self.vector_store = self._initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Initialize QA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info("ArxivLearningAgent initialized successfully")
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing Chroma vector store."""
        try:
            # Try to load existing vector store
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # Test if it has any documents
            if vector_store._collection.count() > 0:
                logger.info(f"Loaded existing knowledge base with {vector_store._collection.count()} documents")
            else:
                logger.info("Created new empty knowledge base")
            return vector_store
        except Exception as e:
            logger.info(f"Creating new knowledge base: {e}")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain for answering questions."""
        prompt_template = """
        You are a helpful research assistant that answers questions based on scientific papers from Arxiv.
        
        Context from papers: {context}
        
        Question: {question}
        
        Instructions:
        - If you can answer the question using the provided context, give a detailed and accurate answer
        - If the context doesn't contain enough information to answer the question, respond with: "I need to learn about it. Please use the learn function to download relevant papers on this topic."
        - When recommending papers, always include the arxiv ID and a brief summary
        - Distinguish between theoretical papers and code-oriented/practical papers when relevant
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def learn_topic(self, topic: str, max_papers: int = 5) -> Dict[str, any]:
        """
        Function 1: Learn about a topic by downloading papers from Arxiv.
        
        Args:
            topic: The topic to learn about
            max_papers: Maximum number of papers to download
            
        Returns:
            Dictionary with learning results
        """
        logger.info(f"Learning about topic: {topic}")
        
        try:
            # Initialize Arxiv retriever
            arxiv_retriever = ArxivRetriever(
                load_max_docs=max_papers,
                get_full_documents=True
            )
            
            # Retrieve papers
            papers = arxiv_retriever.invoke(topic)
            
            if not papers:
                return {
                    "success": False,
                    "message": f"No papers found for topic: {topic}",
                    "papers_processed": 0
                }
            
            # Process and store papers
            all_chunks = []
            paper_info = []
            
            for paper in papers:
                # Add metadata about the topic search
                paper.metadata["search_topic"] = topic
                paper.metadata["learned_date"] = datetime.now().isoformat()
                
                # Split paper into chunks
                chunks = self.text_splitter.split_documents([paper])
                all_chunks.extend(chunks)
                
                # Store paper information
                paper_info.append({
                    "title": paper.metadata.get("Title", "Unknown"),
                    "authors": paper.metadata.get("Authors", "Unknown"),
                    "arxiv_id": paper.metadata.get("Entry ID", "").split("/")[-1] if paper.metadata.get("Entry ID") else "Unknown",
                    "published": paper.metadata.get("Published", "Unknown")
                })
            
            # Add to vector store
            self.vector_store.add_documents(all_chunks)
            
            return {
                "success": True,
                "message": f"Successfully learned about '{topic}'",
                "papers_processed": len(papers),
                "chunks_created": len(all_chunks),
                "papers": paper_info
            }
            
        except Exception as e:
            logger.error(f"Error learning about topic {topic}: {e}")
            return {
                "success": False,
                "message": f"Error learning about topic: {str(e)}",
                "papers_processed": 0
            }
    
    def ask_question(self, question: str) -> Dict[str, any]:
        """
        Function 2: Answer questions using RAG on stored knowledge.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with the answer and source information
        """
        logger.info(f"Answering question: {question}")
        
        try:
            # Check if we have any knowledge
            if self.vector_store._collection.count() == 0:
                return {
                    "answer": "I need to learn about it. My knowledge base is empty. Please use the learn function to download relevant papers first.",
                    "sources": [],
                    "confidence": "no_knowledge"
                }
            
            # Get answer from QA chain
            result = self.qa_chain.invoke({"query": question})
            
            # Process source documents
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "title": doc.metadata.get("Title", "Unknown"),
                    "arxiv_id": doc.metadata.get("Entry ID", "").split("/")[-1] if doc.metadata.get("Entry ID") else "Unknown",
                    "authors": doc.metadata.get("Authors", "Unknown"),
                    "relevance_chunk": doc.page_content[:200] + "..."
                })
            
            return {
                "answer": result["result"],
                "sources": sources,
                "confidence": "with_sources" if sources else "general_knowledge"
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Sorry, I encountered an error while trying to answer your question: {str(e)}",
                "sources": [],
                "confidence": "error"
            }
    
    def explore_knowledge(self) -> Dict[str, any]:
        """
        Function 3: Explore what topics the system knows about.
        
        Returns:
            Dictionary with information about stored knowledge
        """
        logger.info("Exploring available knowledge")
        
        try:
            collection = self.vector_store._collection
            total_docs = collection.count()
            
            if total_docs == 0:
                return {
                    "total_documents": 0,
                    "topics": [],
                    "papers": [],
                    "message": "Knowledge base is empty. Use the learn function to add papers."
                }
            
            # Get all documents metadata (simplified approach)
            results = collection.get(include=["metadatas"])
            
            # Extract unique topics and papers
            topics = set()
            papers = {}
            
            for metadata in results["metadatas"]:
                if metadata and "search_topic" in metadata:
                    topics.add(metadata["search_topic"])
                
                if metadata and "Title" in metadata:
                    arxiv_id = metadata.get("Entry ID", "").split("/")[-1] if metadata.get("Entry ID") else "Unknown"
                    if arxiv_id not in papers:
                        papers[arxiv_id] = {
                            "title": metadata.get("Title", "Unknown"),
                            "authors": metadata.get("Authors", "Unknown"),
                            "published": metadata.get("Published", "Unknown"),
                            "search_topic": metadata.get("search_topic", "Unknown")
                        }
            
            return {
                "total_documents": total_docs,
                "total_papers": len(papers),
                "topics": sorted(list(topics)),
                "papers": list(papers.values()),
                "message": f"Knowledge base contains {len(papers)} papers across {len(topics)} topics"
            }
            
        except Exception as e:
            logger.error(f"Error exploring knowledge: {e}")
            return {
                "total_documents": 0,
                "topics": [],
                "papers": [],
                "message": f"Error exploring knowledge: {str(e)}"
            } 