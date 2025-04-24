import logging
import time
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.model import get_llm
from utils.config import DEFAULT_RETRIEVAL_K

# Configure logging
logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    template="""You are an intelligent and helpful assistant. Your task is to answer questions **strictly** using the information provided in the YouTube video transcript below.

Here is the transcript context:
---------------------
{context}
---------------------

Question: {question}

Instructions:
- Use only the context above to answer the question.
- If the context does **not** contain enough information, reply: "I don't have enough information from the video to answer this question."
- Never guess or add information not present in the context.
- Be concise and relevant.
- If applicable, include **timestamps** mentioned in the context to help locate the answer in the video.

Answer:""",
    input_variables=["context", "question"]
)


def format_docs(retrieved_docs: List[Document]) -> str:
    """Format retrieved documents into a single string with timestamps"""
    formatted_docs = []
    
    for doc in retrieved_docs:
        content = doc.page_content
        metadata = doc.metadata or {}
        timestamp = metadata.get('timestamp', '')
        
        if timestamp:
            formatted_docs.append(f"[Timestamp: {timestamp}]\n{content}")
        else:
            formatted_docs.append(content)
    
    return "\n\n".join(formatted_docs)

def create_qa_chain(vector_store, k=DEFAULT_RETRIEVAL_K, threshold=None):
    """Create a question answering chain with the provided vector store"""
    logger.info(f"Creating QA chain with k={k}, threshold={threshold}")
    
    try:
        # Configure retriever
        search_kwargs = {"k": k}
        if threshold is not None:
            search_kwargs["score_threshold"] = threshold
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # Get LLM
        llm = get_llm()
        
        # Build QA chain
        qa_chain = (
            RunnableParallel(
                {"context": retriever | RunnableLambda(format_docs), 
                 "question": RunnablePassthrough()}
            )
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )
        
        logger.info("QA chain created successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise

class VideoQA:
    """Video question answering manager class"""
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.qa_chain = None
        self._current_k = None
        self._current_threshold = None
        self._query_history = []
        self._max_history_size = 10
        
    def _ensure_qa_chain(self, k=DEFAULT_RETRIEVAL_K, threshold=None):
        """Ensure QA chain is initialized with the current parameters"""
        if (self.qa_chain is None or 
            self._current_k != k or 
            self._current_threshold != threshold):
            self.qa_chain = create_qa_chain(self.vector_store, k=k, threshold=threshold)
            self._current_k = k
            self._current_threshold = threshold
        
    def answer_question(self, question: str, k: int = DEFAULT_RETRIEVAL_K, threshold: Optional[float] = None) -> str:
        """Answer a question about the video"""
        try:
            start_time = time.time()
            logger.info(f"Processing question: {question}")
            
            # Ensure chain is initialized
            self._ensure_qa_chain(k, threshold)
            
            # Generate answer
            answer = self.qa_chain.invoke(question)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Question answered in {elapsed_time:.2f} seconds")
            
            # Record in history
            self._add_to_history(question, answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "Sorry, I encountered an error while trying to answer your question."
            
    def update_retrieval_params(self, k: int = DEFAULT_RETRIEVAL_K, threshold: Optional[float] = None) -> None:
        """Update retrieval parameters for the QA chain"""
        try:
            logger.info(f"Updating QA chain with k={k}, threshold={threshold}")
            self.qa_chain = create_qa_chain(self.vector_store, k=k, threshold=threshold)
            self._current_k = k
            self._current_threshold = threshold
            
        except Exception as e:
            logger.error(f"Error updating QA chain: {str(e)}")
            raise
    
    def _add_to_history(self, question: str, answer: str) -> None:
        """Add a Q&A pair to history"""
        self._query_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Maintain history size
        if len(self._query_history) > self._max_history_size:
            self._query_history.pop(0)
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get the query history"""
        return self._query_history
    
    def search_similar_questions(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar questions in history"""
        query = query.lower()
        scores = []
        
        for entry in self._query_history:
            question = entry["question"].lower()
            # Calculate similarity score
            words1 = set(question.split())
            words2 = set(query.split())
            overlap = len(words1.intersection(words2))
            score = overlap / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
            scores.append((score, entry))
        
        # Sort by score (highest first)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return top k results
        return [item[1] for item in scores[:top_k]]