#!/usr/bin/env python3
"""
Gemma 4 AutoRound W4A16 OpenAI-compatible inference server.
Supports vision input (image_url) via AutoProcessor.
Direct load path: Gemma4UnifiedForConditionalGeneration + convert_hf_model + post_init
Port 8000 — /v1/chat/completions (OpenAI Vision API format)
"""
import base64
import io
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union

MODEL_PATH = "/model"
MODEL_NAME = "gemma4"

app = FastAPI()
model = None
processor = None


def _patch_backend():
    """Override backend selection: skip gptqmodel (incompatible), use tritonv2_zp."""
    try:
        from auto_round.inference import backend as _b
        _orig = _b.get_highest_priority_backend
        def _patched(qcfg, device, packing_format):
            r = _orig(qcfg, device, packing_format)
            return 'auto_round:tritonv2_zp' if (r and 'gptqmodel' in r) else r
        _b.get_highest_priority_backend = _patched
        print("[INFO] Backend: auto_round:tritonv2_zp")
    except Exception as e:
        print(f"[WARN] Backend patch skipped: {e}")


def load_model():
    global model, processor
    _patch_backend()

    from transformers import AutoProcessor

    # Use same class as Gemma4AutoRoundInference (brain_node's tested path)
    try:
        from transformers import Gemma4UnifiedForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelClass

    print("[INFO] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("[INFO] Loading model (bfloat16 → W4A16 via convert_hf_model)...")
    model = ModelClass.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )

    # Same two-step as Gemma4AutoRoundInference: replace + init kernels
    print("[INFO] Applying convert_hf_model + post_init...")
    try:
        from auto_round.inference.convert_model import convert_hf_model, post_init
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, used_backends = convert_hf_model(model, target_device=target_device)
        post_init(model, used_backends)
        print(f"[INFO] Backends used: {used_backends}")
    except Exception as e:
        print(f"[WARN] convert_hf_model failed: {e}")

    model.eval()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[INFO] Model ready. GPU: {mem_gb:.1f}GB")


# ── Pydantic models ────────────────────────────────────────────────────────────

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


# ── Image helpers ──────────────────────────────────────────────────────────────

def _b64_to_pil(url: str):
    from PIL import Image
    _, data = url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _extract_images(raw_messages: list) -> list:
    images = []
    for msg in raw_messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        images.append(_b64_to_pil(url))
    return images


def _to_processor_format(raw_messages: list) -> list:
    out = []
    for msg in raw_messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
        else:
            new_content = []
            for item in content:
                t = item.get("type", "")
                if t == "image_url":
                    new_content.append({"type": "image"})
                elif t == "text":
                    new_content.append({"type": "text", "text": item["text"]})
            if new_content:
                out.append({"role": msg["role"], "content": new_content})
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model",
                  "created": int(time.time()), "owned_by": "local"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_messages = []
    for m in req.messages:
        if isinstance(m.content, str):
            raw_messages.append({"role": m.role, "content": m.content})
        else:
            raw_messages.append({
                "role": m.role,
                "content": [item.model_dump() for item in m.content],
            })

    images = _extract_images(raw_messages)
    proc_messages = _to_processor_format(raw_messages)

    try:
        text = processor.apply_chat_template(
            proc_messages, add_generation_prompt=True, tokenize=False)
        if images:
            inputs = processor(text=text, images=images, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    input_len = inputs["input_ids"].shape[-1]
    max_new = min(req.max_tokens or 512, 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=req.temperature if (req.temperature or 0) > 0 else 1.0,
            top_p=req.top_p or 0.9,
            do_sample=(req.temperature or 0) > 0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_len:]
    text_out = processor.decode(generated, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text_out},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_len,
                  "completion_tokens": len(generated),
                  "total_tokens": input_len + len(generated)},
    }


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
