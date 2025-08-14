import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any, Literal
import pickle
import json
import uvicorn
from pyngrok import ngrok, conf
import nest_asyncio
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel
import torch
from PIL import Image
import io
from io import BytesIO
import logging
import time
import base64
import threading
import asyncio

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading Configuration ---
# Note: In a container, you'd typically download these or have them in the image.
# For this example, we assume the paths are valid within the container's environment.
local_model_path = "./models/medgemma" # Placeholder path for Docker
medsiglip_local_model_path = "./models/medsiglip" # Placeholder path for Docker
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Pydantic Models ---
class ChatMessageContentText(BaseModel):
    type: Literal["text"]
    text: str

class ChatMessageContentImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: Dict[str, str]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Union[ChatMessageContentText, ChatMessageContentImageURL]]]

class OpenAIChatCompletionRequest(BaseModel):
    model: str = "medgemma-4b-it"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=500, ge=1, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream partial progress")

class EmbedTextRequest(BaseModel):
    texts: List[str]

# --- Model Loading ---
# Using placeholder variables for models and processors
# The actual loading will be attempted inside the main block
medgemma_model = None
medgemma_processor = None
medsiglip_model = None
medsiglip_processor = None

def load_models():
    global medgemma_model, medgemma_processor, medsiglip_model, medsiglip_processor
    try:
        logger.info(f"Loading MedGemma model from {local_model_path}...")
        medgemma_model = AutoModelForImageTextToText.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        medgemma_processor = AutoProcessor.from_pretrained(local_model_path)
        logger.info(f"MedGemma model and processor loaded successfully on device: {device}.")
    except Exception as e:
        logger.error(f"Failed to load MedGemma model from {local_model_path}: {e}. This might be expected if model files are not present.")

    try:
        logger.info(f"Loading MedSiglip model from {medsiglip_local_model_path}...")
        medsiglip_model = AutoModel.from_pretrained(
            medsiglip_local_model_path,
            device_map=device
        )
        medsiglip_processor = AutoProcessor.from_pretrained(medsiglip_local_model_path)
        logger.info(f"MedSiglip model and processor loaded successfully on device: {device}.")
    except Exception as e:
        logger.error(f"Failed to load MedSiglip model from {medsiglip_local_model_path}: {e}. This might be expected if model files are not present.")


# Initialize FastAPI
app = FastAPI(
    title="MedGemma Medical AI API (OpenAI-Compatible)",
    description="API for medical image analysis and text queries using MedGemma, with OpenAI-like endpoint structure.",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Load models on startup
    load_models()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Provides a health check for the API."""
    return {
        "status": "healthy",
        "device": device,
        "medgemma_loaded": medgemma_model is not None,
        "medsiglip_loaded": medsiglip_model is not None
    }

# --- OpenAI-like Chat Completions Endpoint (MedGemma) ---
@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest):
    """
    Handles chat completions, supporting both text-only and multimodal (image + text) inputs,
    with an OpenAI-like request and response structure using MedGemma.
    """
    start_time = time.time()
    logger.info(f"Received MedGemma chat completion request for model: {request.model}")

    if medgemma_model is None or medgemma_processor is None:
        raise HTTPException(status_code=503, detail="MedGemma model is not loaded. Please check model paths and logs.")

    messages_for_processor = []
    has_image_input = False
    pil_image = None

    for msg in request.messages:
        processed_content = []
        if isinstance(msg.content, str):
            processed_content.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, ChatMessageContentText):
                    processed_content.append({"type": "text", "text": item.text})
                elif isinstance(item, ChatMessageContentImageURL):
                    image_url = item.image_url.get("url", "")
                    if image_url.startswith("data:image/"):
                        try:
                            base64_data = image_url.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            # The processor expects the image object itself, not a dict
                            processed_content.append(pil_image)
                            has_image_input = True
                        except Exception as e:
                            logger.error(f"Failed to decode base64 image for MedGemma: {e}")
                            raise HTTPException(status_code=400, detail="Invalid base64 image data.")
                    else:
                        logger.warning(f"Unsupported image URL format for MedGemma: {image_url}")
                        raise HTTPException(status_code=400, detail="Only base64 encoded images are supported.")
        messages_for_processor.append({"role": msg.role, "content": processed_content})
    
    # The processor's chat template expects a flat list for content
    # Let's extract the text and image from the user message
    user_prompt = ""
    final_image = None
    for msg in messages_for_processor:
        if msg['role'] == 'user':
            for content_part in msg['content']:
                if isinstance(content_part, str) or (isinstance(content_part, dict) and content_part.get('type') == 'text'):
                     user_prompt += content_part if isinstance(content_part, str) else content_part['text']
                elif isinstance(content_part, Image.Image):
                    final_image = content_part

    if not user_prompt:
        raise HTTPException(status_code=400, detail="A user text prompt is required.")

    inputs = medgemma_processor(text=user_prompt, images=final_image, return_tensors="pt").to(device)

    try:
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = medgemma_model.generate(**inputs, max_new_tokens=request.max_tokens)
        decoded_content = medgemma_processor.decode(generation[0], skip_special_tokens=True)

        processing_time = time.time() - start_time
        logger.info(f"MedGemma request processed in {processing_time:.2f} seconds.")

        response_payload = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": decoded_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generation[0]) - input_len,
                "total_tokens": len(generation[0])
            }
        }
        return JSONResponse(response_payload)

    except Exception as e:
        logger.error(f"Error during MedGemma model generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --------- Image Embedding Endpoint ---------
@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    if medsiglip_model is None or medsiglip_processor is None:
        raise HTTPException(status_code=503, detail="MedSiglip model is not loaded.")
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        inputs = medsiglip_processor(images=[image], return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = medsiglip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features[0].cpu().numpy().tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

# --------- Text Embedding Endpoint ---------
@app.post("/embed/text")
async def embed_text(request: EmbedTextRequest):
    if medsiglip_model is None or medsiglip_processor is None:
        raise HTTPException(status_code=503, detail="MedSiglip model is not loaded.")
    try:
        inputs = medsiglip_processor(text=request.texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            text_features = medsiglip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        embeddings = text_features.cpu().numpy().tolist()

        return {"embeddings": embeddings}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding error: {str(e)}")

if __name__ == "__main__":
    # This block allows running the app directly with uvicorn for local development
    # The Docker container will use the CMD instruction instead.
    uvicorn.run(app, host="0.0.0.0", port=8000)
