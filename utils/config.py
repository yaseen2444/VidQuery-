import os
import tempfile

# Model Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_TOKEN = os.environ.get("HF_TOKEN", "enter token here")

# Retrieval Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_RETRIEVAL_K = 5
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_TEMPERATURE = 0.9

# Project Paths
TEMP_DIR = os.path.join(tempfile.gettempdir(), "youtube_qa_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# FAISS DB Path
FAISS_DB_PATH = os.path.join(TEMP_DIR, "local_faiss_index")

# Model cache path
MODEL_CACHE_PATH = os.path.join(tempfile.gettempdir(), "model_cache")
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Translation settings
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "zh", "ja", "ko", "ru", "ar", "hi", 
    "pt", "it", "nl", "tr", "pl", "sv", "da", "fi", "no", "id"
]
DEFAULT_LANGUAGE = "en"
TRANSLATION_CHUNK_SIZE = 4000  # Characters per translation chunk

# Memory optimization
PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
DEFAULT_BATCH_SIZE = 8
MAX_TRANSCRIPT_CHUNKS = 2000  # Limit for very large videos

# Interface settings
MAX_HISTORY_SIZE = 10