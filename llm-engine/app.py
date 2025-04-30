import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/llm-engine.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("llm-engine")

app = FastAPI(
    title="LLM Engine API", description="Local LLM with OpenAI Compatible API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define models for API
class CompletionRequest(BaseModel):
    model: str = Field(default="local")
    prompt: str
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    stop: Optional[List[str]] = None
    stream: bool = Field(default=False)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="local")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    stop: Optional[List[str]] = None
    stream: bool = Field(default=False)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage


# Load model based on environment variables

# MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"

MODEL_TYPE = os.environ.get("MODEL_TYPE", "llama")
USE_GPU = torch.cuda.is_available()

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    logger.error(f"Model path {MODEL_PATH} does not exist!")
    raise ValueError(f"Model path {MODEL_PATH} does not exist!")

logger.info(f"Loading model from {MODEL_PATH} with type {MODEL_TYPE}")
logger.info(f"Using GPU: {USE_GPU}")

# Enhanced GPU configuration
if USE_GPU:
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Set stronger GPU memory efficiency settings
    torch.cuda.empty_cache()
    device_map = "auto"
    torch_dtype = torch.float16  # Use FP16 for better GPU memory efficiency
else:
    device_map = None
    torch_dtype = torch.float32

# Initialize model based on format (GGUF, HF, etc.)
if MODEL_PATH.endswith(".gguf"):
    logger.info("Loading GGUF model with transformers")

    # For GGUF models, we'll use llama-cpp-python through the transformers LlamaTokenizer
    try:
        from llama_cpp import Llama

        # Initialize Llama with GPU support
        n_gpu_layers = -1 if USE_GPU else 0  # -1 means use all available layers on GPU

        # Load the model
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
        )

        # Configure tokenizer for token counting
        if MODEL_TYPE == "llama":
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        elif MODEL_TYPE == "mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

        logger.info("GGUF model loaded with llama-cpp-python")

        # We'll use a custom pipeline approach for GGUF models
        model = None
        pipe = None

    except ImportError:
        logger.error("llama-cpp-python is required to load GGUF models. Please install it with 'pip install llama-cpp-python'")
        raise ImportError("llama-cpp-python is required to load GGUF models")
else:
    logger.info("Loading model with HuggingFace Transformers")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # Create the pipeline with GPU support
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if USE_GPU else -1,
        torch_dtype=torch_dtype
    )

logger.info("Model loaded successfully!")


# Helper for token counting
def count_tokens(text: str) -> int:
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough estimate for GGUF models
        return len(text.split()) * 1.5


# API Endpoints
@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    try:
        logger.info(f"Received completion request: {request.prompt[:50]}...")

        if MODEL_PATH.endswith(".gguf"):
            # Use llama-cpp-python for GGUF models
            response = llm(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop or [],
            )["choices"][0]["text"]
        else:
            # HuggingFace pipeline approach for other models
            generation = pipe(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                return_full_text=False,
            )
            response = generation[0]["generated_text"]

        prompt_tokens = count_tokens(request.prompt)
        completion_tokens = count_tokens(response)

        return CompletionResponse(
            id=f"cmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "text": response,
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                }
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info(
            f"Received chat completion request with {len(request.messages)} messages"
        )

        # Format chat messages into prompt
        if MODEL_TYPE == "llama" or MODEL_TYPE == "mistral":
            # Llama/Mistral chat template
            formatted_prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    formatted_prompt += f"<|system|>\n{msg.content}</s>\n"
                elif msg.role == "user":
                    formatted_prompt += f"<|user|>\n{msg.content}</s>\n"
                elif msg.role == "assistant":
                    formatted_prompt += f"<|assistant|>\n{msg.content}</s>\n"

            formatted_prompt += "<|assistant|>\n"
        else:
            # Generic chat template
            formatted_prompt = ""
            for msg in request.messages:
                formatted_prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            formatted_prompt += "Assistant: "

        # Generate completion
        if MODEL_PATH.endswith(".gguf"):
            # Use llama-cpp-python for GGUF models
            response = llm(
                formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop or ["</s>", "User:", "System:"],
            )["choices"][0]["text"]
        else:
            # HuggingFace pipeline approach for other models
            generation = pipe(
                formatted_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )
            response = generation[0]["generated_text"][len(formatted_prompt):]

        # Strip any stop tokens
        if request.stop:
            for stop_seq in request.stop:
                if response.endswith(stop_seq):
                    response = response[: -len(stop_seq)]

        prompt_tokens = count_tokens(formatted_prompt)
        completion_tokens = count_tokens(response)

        return ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_PATH, "gpu": USE_GPU}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")
