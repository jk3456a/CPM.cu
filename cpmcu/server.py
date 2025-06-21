#!/usr/bin/env python3
import os
import sys
import argparse
import asyncio
import json
import torch
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

from .llm import LLM
from .llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from .speculative import LLM_with_eagle
from .speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from .api_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ErrorResponse,
    HealthResponse
)
from .utils import (
    load_config_from_file,
    get_default_config,
    check_or_download_model,
    get_model_paths,
    get_minicpm4_yarn_factors,
    create_model,
    apply_minicpm4_yarn_config,
    setup_frspec_vocab
)
from .args import parse_server_args, display_config_summary

# Global model instance
model_instance: Optional[LLM] = None
model_config: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup"""
    global model_instance, model_config
    
    print(f"Loading model with configuration:")
    print(f"Configuration: {model_config['config']}")
    
    try:
        config = model_config['config']
        
        # Get model paths directly here
        eagle_path, base_path, eagle_repo_id, base_repo_id = get_model_paths(config['path_prefix'], config)
        
        print(f"Eagle model path: {eagle_path}")
        print(f"Base model path: {base_path}")
        
        # Create model instance
        model_instance = create_model(eagle_path, base_path, config)
        
        # Initialize model storage
        model_instance.init_storage()
        
        # Apply MiniCPM4 YARN configuration if enabled
        if config['test_minicpm4'] and config['minicpm4_yarn']:
            print("Applying MiniCPM4 YARN rope_scaling parameters")
            apply_minicpm4_yarn_config(model_instance, config)
        
        # Load frequency speculative vocabulary if enabled
        if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
            print(f"Loading frequency vocabulary...")
            setup_frspec_vocab(model_instance, config, eagle_path, eagle_repo_id)
        
        # Load model weights
        model_instance.load_from_hf()
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    print("Shutting down...")
    model_instance = None

# Create FastAPI app
app = FastAPI(
    title="CPM.cu OpenAI API Server",
    description="OpenAI API compatible server for CPM.cu models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_messages_to_prompt(messages: list, tokenizer) -> str:
    """Convert OpenAI messages format to a single prompt string using chat template"""
    
    # Convert OpenAI messages format to the format expected by chat template
    chat_messages = []
    
    for message in messages:
        chat_messages.append({
            "role": message.role,
            "content": message.content
        })
    
    # Use tokenizer's chat template to format the prompt
    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        # Fallback to simple formatting if chat template fails
        print(f"Warning: Chat template failed ({e}), falling back to simple formatting")
        prompt_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)

def get_stop_tokens(stop_param: Optional[Union[str, list]] = None) -> list:
    """Convert stop parameter to token IDs"""
    if not stop_param:
        return []
    
    if isinstance(stop_param, str):
        stop_list = [stop_param]
    else:
        stop_list = stop_param
    
    # Convert to token IDs using tokenizer
    stop_token_ids = []
    for stop_str in stop_list:
        try:
            tokens = model_instance.tokenizer.encode(stop_str, add_special_tokens=False)
            stop_token_ids.extend(tokens)
        except:
            pass
    
    return stop_token_ids

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        model_loaded=model_instance is not None,
        memory_usage=f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else None
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI Chat Completions API endpoint"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get tokenizer from model instance first
        tokenizer = getattr(model_instance, 'tokenizer', None)
        if hasattr(model_instance, 'base_model') and hasattr(model_instance.base_model, 'tokenizer'):
            tokenizer = model_instance.base_model.tokenizer
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not available")
        
        # Format messages to prompt using chat template
        prompt = format_messages_to_prompt(request.messages, tokenizer)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(torch.int32).cuda()
        
        # Get stop token IDs and add EOS token if use_terminators is enabled
        stop_tokens = get_stop_tokens(request.stop)
        config = model_config['config']
        if config.get('use_terminators', True):
            if tokenizer.eos_token_id not in stop_tokens:
                stop_tokens.append(tokenizer.eos_token_id)
        
        # Set temperature for this request
        original_temp = getattr(model_instance, 'temperature', 0.0)
        model_instance.temperature = request.temperature or 0.0
        
        try:
            if request.stream:
                return StreamingResponse(
                    stream_chat_completion(
                        input_ids, 
                        request.max_tokens or 100,
                        stop_tokens,
                        request.model,
                        request,
                        tokenizer
                    ),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    }
                )
            else:
                return await generate_chat_completion(
                    input_ids,
                    request.max_tokens or 100, 
                    stop_tokens,
                    request.model,
                    request,
                    tokenizer
                )
        finally:
            # Restore original temperature
            model_instance.temperature = original_temp
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def generate_chat_completion(
    input_ids: torch.Tensor, 
    max_tokens: int, 
    stop_tokens: list,
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> ChatCompletionResponse:
    """Generate non-streaming chat completion"""
    
    # Generate tokens using the same interface as test_generate.py
    gen_result = model_instance.generate(
        input_ids.view(-1), 
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=False
    )
    
    # Handle different return formats based on model type
    config = model_config['config']
    if config.get('apply_eagle', False):
        # Eagle models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = gen_result
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = gen_result
    
    # Decode response
    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Extract only the assistant's response (after the last "Assistant:")
    if "Assistant:" in generated_text:
        response_text = generated_text.split("Assistant:")[-1].strip()
    else:
        response_text = generated_text.strip()
    
    # Determine finish reason
    finish_reason = "stop"
    if len(tokens) >= max_tokens:
        finish_reason = "length"
    elif any(token in tokens for token in stop_tokens):
        finish_reason = "stop"
    
    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=finish_reason
            )
        ],
        usage={
            "prompt_tokens": input_ids.numel(),
            "completion_tokens": len(tokens),
            "total_tokens": input_ids.numel() + len(tokens)
        }
    )

async def stream_chat_completion(
    input_ids: torch.Tensor,
    max_tokens: int,
    stop_tokens: list, 
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion"""
    
    import time
    import uuid
    import asyncio
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Start generation stream
    stream_gen = model_instance.generate(
        input_ids.view(-1),
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=True
    )
    
    accumulated_text = ""
    
    try:
        for chunk in stream_gen:
            token = chunk['token']
            text = chunk['text']
            is_finished = chunk['is_finished']
            
            # Only send the new text part
            accumulated_text += text
            
            # Extract only assistant's response
            if "Assistant:" in accumulated_text:
                display_text = accumulated_text.split("Assistant:")[-1].strip()
            else:
                display_text = accumulated_text.strip()
            
            # Create streaming response
            if not is_finished:
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": text},
                            finish_reason=None
                        )
                    ]
                )
            else:
                # Final chunk
                finish_reason = "stop"
                if token in stop_tokens:
                    finish_reason = "stop"
                elif len(accumulated_text) >= max_tokens:
                    finish_reason = "length"
                
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason=finish_reason
                        )
                    ]
                )
            
            # Yield with immediate flush to ensure real-time streaming
            yield f"data: {response.model_dump_json()}\n\n"
            
            # Add a small async yield to allow the event loop to process
            # This ensures immediate delivery of each chunk
            await asyncio.sleep(0)
            
            if is_finished:
                break
                
    except Exception as e:
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "generation_failed"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"
    await asyncio.sleep(0)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": str(exc),
                "type": "internal_error", 
                "code": "server_error"
            }
        ).model_dump()
    )

def main():
    """Server entry point using unified argument processing"""
    from .args import parse_server_args, display_config_summary
    from .utils import get_minicpm4_yarn_factors
    
    # Use unified argument parsing
    args, config = parse_server_args()
    
    # Server-specific configuration adjustments
    if not config['test_minicpm4']:
        print("test_minicpm4 is False, set apply_sparse to False")
        config['apply_sparse'] = False
    
    # Process MiniCPM4 YARN configuration if enabled
    if config['test_minicpm4'] and config['minicpm4_yarn']:
        print("Adding MiniCPM4 YARN configuration...")
        config['yarn_factors'] = get_minicpm4_yarn_factors()
    
    # Set global model config
    global model_config
    model_config = {"config": config}
    
    # Display configuration summary
    display_config_summary(config, "Server Configuration")
    
    print(f"Starting CPM.cu OpenAI API Server on {config.get('host', '0.0.0.0')}:{config.get('port', 8000)}")
    
    uvicorn.run(
        app,
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        log_level="info"
    )

if __name__ == "__main__":
    main() 