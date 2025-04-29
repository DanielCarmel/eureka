import os
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Union, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            "logs/llm-engine.log", maxBytes=10485760, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("llm-engine")

app = FastAPI(title="LLM Engine API", description="Local LLM with OpenAI Compatible API")

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
MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "llama")
USE_GPU = torch.cuda.is_available()

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    logger.error(f"Model path {MODEL_PATH} does not exist!")
    raise ValueError(f"Model path {MODEL_PATH} does not exist!")

logger.info(f"Loading model from {MODEL_PATH} with type {MODEL_TYPE}")
logger.info(f"Using GPU: {USE_GPU}")

# Initialize model based on format (GGUF, HF, etc.)
if MODEL_PATH.endswith(".gguf"):
    logger.info("Loading GGUF model with CTransformers")
    model = CTAutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type=MODEL_TYPE,
        gpu_layers=64 if USE_GPU else 0
    )
    tokenizer = None
else:
    logger.info("Loading model with HuggingFace Transformers")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto" if USE_GPU else None,
        torch_dtype=torch.float16 if USE_GPU else torch.float32,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if USE_GPU else -1
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
        
        if hasattr(model, "generate"):
            # GGUF model approach
            response = model(
                request.prompt, 
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop
            )
        else:
            # HuggingFace pipeline approach
            generation = pipe(
                request.prompt,
                max_length=len(request.prompt.split()) + request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                stopping_criteria=request.stop if request.stop else None
            )
            response = generation[0]["generated_text"][len(request.prompt):]
        
        prompt_tokens = count_tokens(request.prompt)
        completion_tokens = count_tokens(response)
        
        return CompletionResponse(
            id=f"cmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "text": response,
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None
            }],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info(f"Received chat completion request with {len(request.messages)} messages")
        
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
        if hasattr(model, "generate"):
            # GGUF model approach
            response = model(
                formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop or ["</s>", "User:", "System:"]
            )
        else:
            # HuggingFace pipeline approach
            generation = pipe(
                formatted_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
            response = generation[0]["generated_text"][len(formatted_prompt):]
        
        # Strip any stop tokens
        if request.stop:
            for stop_seq in request.stop:
                if response.endswith(stop_seq):
                    response = response[:-len(stop_seq)]
        
        prompt_tokens = count_tokens(formatted_prompt)
        completion_tokens = count_tokens(response)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
                "index": 0
            }],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_PATH, "gpu": USE_GPU}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")