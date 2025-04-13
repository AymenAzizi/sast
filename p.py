"""
AI model management for vulnerability fix generation.
"""

import os
import logging
import requests
import json
import time

# Check for transformers availability
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
from ai_module import get_current_model, HF_API_TOKEN, GENERATION_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

# Global variables for model caching
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """
    Get or load the model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer)
    """
    global _model, _tokenizer
    
    # Return cached model if available
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not available, using API fallback")
        return None, None
    
    model_name = get_current_model()
    
    try:
        logger.info(f"Loading model {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check for CUDA availability
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        
        # Load model
        if device == "cuda":
            # Load model on GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,  # Use 8-bit quantization to reduce memory usage
                torch_dtype="auto"
            )
        else:
            # Load model on CPU
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        _model = model
        _tokenizer = tokenizer
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def generate_text(prompt, max_tokens=512):
    """
    Generate text using the local model.
    
    Args:
        prompt (str): Input prompt
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
    
    Returns:
        str: Generated text
    """
    model, tokenizer = get_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        logger.warning("Using Hugging Face API fallback due to model loading failure")
        return generate_text_api(prompt, max_tokens)
    
    try:
        # Create generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False
        )
        
        # Generate text
        outputs = generator(
            prompt,
            max_new_tokens=max_tokens,
            **GENERATION_CONFIG
        )
        
        # Extract generated text
        generated_text = outputs[0]['generated_text']
        
        return generated_text
    
    except Exception as e:
        logger.error(f"Error generating text with local model: {str(e)}")
        logger.warning("Falling back to Hugging Face API")
        return generate_text_api(prompt, max_tokens)

def generate_text_api(prompt, max_tokens=512):
    """
    Generate text using the Hugging Face API.
    
    Args:
        prompt (str): Input prompt
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
    
    Returns:
        str: Generated text
    """
    model_name = get_current_model()
    
    # API URL for Hugging Face Inference API
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    # API headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add authorization header if token is available
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    # API payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
            "top_k": GENERATION_CONFIG["top_k"],
            "repetition_penalty": GENERATION_CONFIG["repetition_penalty"],
            "do_sample": GENERATION_CONFIG["do_sample"],
            "return_full_text": False
        }
    }
    
    # Make API request
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            
            # Check if the model is still loading
            if response.status_code == 503:
                estimated_time = json.loads(response.content.decode())
                wait_time = estimated_time.get("estimated_time", retry_delay)
                logger.info(f"Model is loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
            
            # If response format is different
            return str(result)
        
        except Exception as e:
            logger.error(f"API request attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("All API request attempts failed")
                return ""
    
    return ""

def extract_code_block(text):
    """
    Extract code block from generated text.
    
    Args:
        text (str): Text containing code blocks
    
    Returns:
        str: Extracted code or original text if no code block found
    """
    # Find code blocks delimited by triple backticks
    import re
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Return the first code block
        return code_blocks[0].strip()
    
    return text  # Return original text if no code block found
