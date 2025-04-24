import streamlit as st
import re
import os
import logging
import torch
import gc
import time
from utils.transcribe import fetch_transcript
from utils.embedding import preprocess_transcript
from utils.retriever import create_qa_chain, VideoQA
from utils.model import unload_model
from utils.config import PYTORCH_CUDA_ALLOC_CONF, DEFAULT_RETRIEVAL_K
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

def extract_video_id(url):
    """Extract the video ID from a YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\?\/]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def clean_memory():
    """Clean up memory and return current usage statistics"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return {
            "allocated_gb": f"{allocated:.2f}",
            "reserved_gb": f"{reserved:.2f}"
        }
    return {"allocated_gb": "N/A", "reserved_gb": "N/A"}

def get_session_id():
    """Create or get session ID"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def init_session_state():
    """Initialize session state variables"""
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "processing_error" not in st.session_state:
        st.session_state.processing_error = None
    if "memory_stats" not in st.session_state:
        st.session_state.memory_stats = clean_memory()
    if "history" not in st.session_state:
        st.session_state.history = []

def process_video(video_id):
    """Process a YouTube video for QA"""
    try:
        # Step 1: Clean memory
        clean_memory()
        
        # Step 2: Fetch transcript
        logger.info(f"Fetching transcript for video ID: {video_id}")
        transcript = fetch_transcript(video_id)
        st.session_state.transcript = transcript
        
        # Step 3: Clean memory
        clean_memory()
        
        # Step 4: Process transcript
        logger.info("Creating embeddings...")
        vector_store = preprocess_transcript(transcript)
        
        if vector_store:
            st.session_state.vector_store = vector_store
            
            # Step 5: Clean memory
            clean_memory()
            
            # Step 6: Create QA system
            logger.info("Creating QA system...")
            st.session_state.qa_system = VideoQA(vector_store)
            
            st.session_state.processed = True
            st.session_state.processing_error = None
            logger.info(f"Successfully processed video ID: {video_id}")
        else:
            st.session_state.processing_error = "Failed to process transcript."
            logger.error(f"Failed to process transcript for video ID: {video_id}")
    
    except Exception as e:
        st.session_state.processing_error = str(e)
        logger.exception(f"Error processing video ID: {video_id}")
    
    finally:
        # Update memory stats
        st.session_state.memory_stats = clean_memory()

def main():
    st.set_page_config(
        page_title="YouTube Video QA System",
        page_icon="🎥",
        layout="wide"
    )
    
    init_session_state()
    get_session_id()
    
    st.title("🎥 YouTube Video Question Answering System")
    st.write("Ask questions about any YouTube video with captions or transcripts")
    
    # Memory usage info in sidebar
    if torch.cuda.is_available():
        st.sidebar.subheader("GPU Memory Usage")
        st.sidebar.text(f"Allocated: {st.session_state.memory_stats['allocated_gb']} GB")
        st.sidebar.text(f"Reserved: {st.session_state.memory_stats['reserved_gb']} GB")
        
        if st.sidebar.button("Refresh Memory Stats"):
            st.session_state.memory_stats = clean_memory()
            st.experimental_rerun()
    
    # Video URL input
    with st.form(key="url_form"):
        url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        process_btn = st.form_submit_button("Process Video")
    
    # Process the video
    if process_btn and url:
        st.session_state.processed = False
        st.session_state.processing_error = None
        
        try:
            # Extract video ID
            with st.spinner("Extracting video ID..."):
                video_id = extract_video_id(url)
                if not video_id:
                    st.error("Could not extract a valid YouTube video ID from the URL.")
                    return
                
                st.session_state.video_id = video_id
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "Preparing to process video...",
                "Fetching transcript...",
                "Creating embeddings...",
                "Building QA system...",
                "Finalizing..."
            ]
            
            for i, stage in enumerate(stages):
                status_text.text(stage)
                progress_bar.progress((i + 1) / len(stages))
                time.sleep(0.5)
            
            # Process video
            with st.spinner("Processing video... This may take a while."):
                process_video(video_id)
            
            # Check result
            if st.session_state.processed:
                progress_bar.progress(100)
                status_text.text("Video processed successfully!")
                st.success("Video processed successfully! You can now ask questions about the content.")
            elif st.session_state.processing_error:
                st.error(f"Error processing video: {st.session_state.processing_error}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.exception(f"Error processing video URL: {url}")
    
    # Display video info
    if st.session_state.processed and st.session_state.video_id:
        st.sidebar.subheader("Video Information")
        st.sidebar.write(f"Video ID: {st.session_state.video_id}")
        
        # Display YouTube video
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        
        transcript_text = st.session_state.transcript
        if isinstance(transcript_text, tuple):
            transcript_text, lang = transcript_text
            st.sidebar.write(f"Language: {lang}")
            
        # Show transcript option
        if st.sidebar.checkbox("Show Transcript"):
            st.sidebar.text_area("Video Transcript", transcript_text, height=300)
    
    # Question answering section
    if st.session_state.processed and st.session_state.qa_system:
        st.subheader("Ask Questions About The Video")
        query = st.text_input("Your Question:", placeholder="What is the main topic discussed in this video?")
        
        col1, col2 = st.columns([1, 3])
        k_value = col1.slider("Number of context chunks:", min_value=1, max_value=10, value=DEFAULT_RETRIEVAL_K)
        
        if query:
            try:
                with st.spinner("Generating answer..."):
                    clean_memory()
                    response = st.session_state.qa_system.answer_question(query, k=k_value)
                    st.session_state.memory_stats = clean_memory()
                
                st.subheader("Answer:")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                logger.exception(f"Error generating answer for query: {query}")
                clean_memory()
        
        # Show history
        if st.session_state.qa_system and st.session_state.qa_system._query_history:
            with st.expander("Question History"):
                for i, qa in enumerate(st.session_state.qa_system._query_history):
                    st.markdown(f"**Q{i+1}: {qa['question']}**")
                    st.markdown(f"A: {qa['answer']}")
                    st.divider()
    
    # Footer and cleanup options
    st.sidebar.markdown("---")
    st.sidebar.info("This app uses YouTube transcripts and the Llama model to answer questions about video content.")
    
    if st.sidebar.button("Clear Session Data"):
        if st.session_state.get("qa_system"):
            try:
                unload_model()
            except Exception as e:
                logger.exception("Error unloading model")
        
        clean_memory()
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        st.success("Session data cleared and memory released")
        st.experimental_rerun()

if __name__ == "__main__":
    main()