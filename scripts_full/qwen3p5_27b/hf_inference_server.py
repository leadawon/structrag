#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server using HuggingFace transformers.
Drop-in replacement for vLLM server for Qwen3.5-27B inference.
"""

import argparse
import json
import time
import uuid
import os
import threading
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list] = None
    stream: Optional[bool] = False


app = FastAPI()
model = None
tokenizer = None
model_lock = threading.Lock()
MAX_MODEL_LEN = 32768


def _context_length_error(prompt_tokens: int, max_new_tokens: int) -> JSONResponse:
    total = prompt_tokens + max_new_tokens
    msg = (
        f"Please reduce the length of the messages or completion so that their total "
        f"length is within the model's context window. "
        f"maximum context length is {MAX_MODEL_LEN} tokens. "
        f"However, you requested {total} tokens "
        f"({prompt_tokens} in the messages, {max_new_tokens} in the completion)."
    )
    return JSONResponse(status_code=400, content={"error": {"message": msg}})


def load_model(model_path: str, dtype: str = "bfloat16", gpu_ids: str = "0,1,2,3"):
    global model, tokenizer
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} (device_map=auto, dtype={dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully.")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i == 0:
            print(f"  first param device: {param.device}")
            break


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": os.environ.get("SERVED_MODEL_NAME", "Qwen3.5-27B"), "object": "model"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [{"role": m["role"] if isinstance(m, dict) else m.role,
                  "content": m["content"] if isinstance(m, dict) else m.content}
                 for m in req.messages]

    try:
        enable_thinking = os.environ.get("STRUCTRAG_ENABLE_THINKING", "0") == "1"
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_tokens = input_ids.shape[1]
    max_new_tokens = req.max_tokens or 2048

    # Reject before touching GPU if the context is already too long
    if prompt_tokens + max_new_tokens > MAX_MODEL_LEN:
        return _context_length_error(prompt_tokens, max_new_tokens)

    input_ids = input_ids.to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    temperature = req.temperature if req.temperature is not None else 0.0
    top_p = req.top_p if req.top_p is not None else 1.0

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    try:
        with model_lock:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _context_length_error(prompt_tokens, max_new_tokens)
    finally:
        torch.cuda.empty_cache()

    new_ids = output_ids[0][prompt_tokens:]
    response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    completion_tokens = new_ids.shape[0]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1225)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    MAX_MODEL_LEN = args.max_model_len
    load_model(args.model, args.dtype, args.gpu_ids)
    uvicorn.run(app, host=args.host, port=args.port)
