import logging
import torch
import gc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator
import uuid
from utils.config import (
    EMBEDDING_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    TRANSLATION_CHUNK_SIZE, MAX_TRANSCRIPT_CHUNKS, DEFAULT_BATCH_SIZE,
    DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
)

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_transcript(transcript_data):
    """Process transcript with memory optimization and multi-language support"""
    logger.info("Starting transcript preprocessing")
    
    # Clear memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Handle case where transcript_data is a tuple (text, language)
    if isinstance(transcript_data, tuple):
        if len(transcript_data) >= 2:
            transcript = transcript_data[0]
            language = transcript_data[1]
        else:
            transcript = transcript_data[0]
            language = DEFAULT_LANGUAGE
    else:
        transcript = transcript_data
        language = DEFAULT_LANGUAGE
    
    if not transcript or not isinstance(transcript, str) or len(transcript.strip()) == 0:
        logger.error("Empty or invalid transcript provided")
        return None
    
    # Translate to English if needed
    if language and language != "en" and language in SUPPORTED_LANGUAGES:
        try:
            logger.info(f"Translating from {language} to English")
            translated_parts = []
            
            for i in range(0, len(transcript), TRANSLATION_CHUNK_SIZE):
                chunk = transcript[i:i+TRANSLATION_CHUNK_SIZE]
                translated_chunk = GoogleTranslator(source=language, target="en").translate(chunk)
                translated_parts.append(translated_chunk)
                gc.collect()
                
            transcript = " ".join(translated_parts)
            logger.info("Translation completed successfully")
        except Exception as e:
            logger.exception(f"Translation error: {e}")
            logger.info("Proceeding with original untranslated text")
    
    try:
        # Split transcript into smaller chunks for processing
        logger.info("Splitting transcript into chunks")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.create_documents([transcript])
        
        # Limit number of chunks for very large videos
        if len(chunks) > MAX_TRANSCRIPT_CHUNKS:
            logger.warning(f"Transcript too large, limiting to {MAX_TRANSCRIPT_CHUNKS} chunks")
            chunks = chunks[:MAX_TRANSCRIPT_CHUNKS]
        
        # Add unique IDs to chunks for retrieval
        for chunk in chunks:
            chunk.metadata["id"] = str(uuid.uuid4())
            
        logger.info(f"Created {len(chunks)} document chunks")
        
        # Process in batches to manage memory
        batch_size = DEFAULT_BATCH_SIZE
        
        # Initialize embeddings
        logger.info(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu", 
                "batch_size": batch_size
            }
        )
        
        # Process documents in batches
        if len(chunks) > batch_size * 5:  # Only batch if there are many chunks
            logger.info(f"Processing {len(chunks)} chunks in batches")
            vector_store = None
            
            for i in range(0, len(chunks), batch_size * 5):
                batch = chunks[i:i+batch_size * 5]
                logger.info(f"Processing batch {i//(batch_size * 5) + 1}/{(len(chunks)-1)//(batch_size * 5) + 1}")
                
                batch_store = FAISS.from_documents(batch, embeddings)
                
                if vector_store is None:
                    vector_store = batch_store
                else:
                    vector_store.merge_from(batch_store)
                
                # Clear memory after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Create vector store all at once for smaller document sets
            logger.info("Creating FAISS vector store")
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        logger.info("Successfully created vector store")
        return vector_store
        
    except Exception as e:
        logger.exception(f"Error in preprocessing transcript: {e}")
        return None