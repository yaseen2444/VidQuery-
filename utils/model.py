import logging
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from utils.config import MODEL_NAME, HF_TOKEN, MODEL_CACHE_PATH, PYTORCH_CUDA_ALLOC_CONF

# Configure logging
logger = logging.getLogger(__name__)

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

def load_model(model_name=MODEL_NAME, use_auth_token=HF_TOKEN):
    """Load the Llama model with memory optimization techniques"""
    logger.info(f"Loading model: {model_name}")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token,
            cache_dir=MODEL_CACHE_PATH
        )
        logger.info("Tokenizer loaded successfully")
        
        # Check available GPU memory if CUDA is available
        device_map = "auto"
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Available GPU memory: {free_memory_gb:.2f} GB")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Prepare model loading arguments with memory optimizations
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "cache_dir": MODEL_CACHE_PATH
        }
        
        if use_auth_token:
            model_kwargs["use_auth_token"] = use_auth_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise

def create_llm_pipeline(model, tokenizer):
    """Create a memory-efficient text generation pipeline"""
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        batch_size=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Global variables for model caching
_llm = None
_model = None
_tokenizer = None

def get_llm():
    """Get or create the LLM instance with memory management"""
    global _model, _tokenizer, _llm
    
    if _llm is None:
        try:
            # Free up memory if possible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache before loading model")
                
            # Load the model
            _model, _tokenizer = load_model()
            _llm = create_llm_pipeline(_model, _tokenizer)
        except Exception as e:
            logger.exception("Failed to initialize model")
            raise e
    
    return _llm

def unload_model():
    """Explicitly unload model to free up GPU memory"""
    global _model, _tokenizer, _llm
    
    if _model is not None:
        del _model
        _model = None
        
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
        
    if _llm is not None:
        del _llm
        _llm = None
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Model unloaded and CUDA cache cleared")