# medgemma_api_server.py

import io
import json
import logging
import re
import time
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- 1. Initialize Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Model Loading Configuration ---
# IMPORTANT: Update this path to where your model is stored.
# Using an environment variable is recommended for production.
LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/your/model/folder")

# --- 3. Load Model and Processor ---
model = None
processor = None
device = "cpu"

try:
    logger.info(f"Loading MedGemma model from {LOCAL_MODEL_PATH}...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Model path does not exist: {LOCAL_MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
    logger.info(f"MedGemma model and processor loaded successfully on device: {device}.")
except Exception as e:
    logger.error(f"FATAL: Failed to load MedGemma model: {e}")
    # We exit here because the application is not usable without the model.
    exit()

# --- 4. Initialize FastAPI App ---
app = FastAPI(
    title="MedGemma Medical AI API",
    description="API for medical image analysis and text queries using MedGemma",
    version="0.1.0"
)

# CORS configuration to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. AI System Prompts ---
# (Prompts remain unchanged)
AI_TEXT_REPORT_JSON_SCHEMA = """
{{
    "overall_confidence": "High/Medium/Low Confidence (e.g., 'High Confidence')",
    "clinical_summary": "Provide a concise 2-3 sentence overview of the patient's presentation and key medical information.",
    "detailed_analysis": "Present key findings from the clinical text, their significance, relevant pathophysiology, risk factors, and prognostic indicators.",
    "differential_considerations": [
        {{"diagnosis": "Most likely diagnosis 1", "reasoning": "Supporting reasoning for diagnosis 1"}},
        {{"diagnosis": "Most likely diagnosis 2", "reasoning": "Supporting reasoning for diagnosis 2"}}
    ],
    "clinical_recommendations": [
        "Immediate actions or necessary assessments.",
        "Appropriate follow-up care and monitoring.",
        "Specialist referrals when appropriate."
    ],
    "patient_education_points": [
        "Clear, concise information relevant to the case for patient understanding point 1.",
        "Point 2."
    ],
    "disclaimer": "This analysis is for educational and informational purposes only. All medical decisions require evaluation by qualified healthcare professionals. This AI cannot replace clinical judgment, physical examination, or comprehensive patient assessment."
}}
"""

AI_IMAGE_REPORT_JSON_SCHEMA = """
{{
    "overall_confidence": "High/Medium/Low Confidence (e.g., 'High Confidence')",
    "clinical_summary": "Provide a concise 2-3 sentence overview of the key patient/image findings.",
    "detailed_analysis": {{
        "technical_assessment": "Comment on image quality, technique, and any limitations that might affect interpretation.",
        "systematic_examination": "Describe normal findings first, then a detailed description of all abnormal findings. Include location, size, morphology.",
        "clinical_correlation": "Relate observed abnormalities to potential clinical significance and disease processes."
    }},
    "differential_considerations": [
        {{"diagnosis": "Most likely diagnosis 1", "reasoning": "Supporting reasoning"}},
        {{"diagnosis": "Most likely diagnosis 2", "reasoning": "Distinguishing signs"}}
    ],
    "clinical_recommendations": [
        "Suggest additional imaging, further diagnostic tests, or clinical correlation.",
        "Outline immediate actions or necessary assessments.",
        "Appropriate follow-up care and monitoring.",
        "Specialist referrals when appropriate."
    ],
    "patient_education_points": [
        "Clear, concise information for patient understanding point 1.",
        "Point 2."
    ],
    "disclaimer": "This AI-generated analysis is intended for educational, research, and preliminary diagnostic support only. All interpretations must be reviewed and confirmed by a qualified radiologist or licensed medical professional before being used in clinical decision-making. Emergency findings should prompt immediate medical attention."
}}
"""

MEDICAL_TEXT_SYSTEM_PROMPT = f"You are MedGemma, an expert medical AI...Your entire response MUST be a single, valid JSON object...JSON Schema:\n```json\n{AI_TEXT_REPORT_JSON_SCHEMA}```..."
MEDICAL_IMAGE_SYSTEM_PROMPT = f"You are MedGemma, an expert medical AI with advanced capabilities in medical image analysis...Your entire response MUST be a single, valid JSON object...JSON Schema:\n```json\n{AI_IMAGE_REPORT_JSON_SCHEMA}```..."
SOAP_NOTE_SYSTEM_PROMPT = """You are a clinical documentation assistant...Provide the SOAP note directly..."""


# --- 6. Pydantic Request Models ---
class TextQuery(BaseModel):
    question: str
    system_prompt: Optional[str] = Field(default=MEDICAL_TEXT_SYSTEM_PROMPT, description="System prompt for the text model.")


# --- 7. Helper Functions ---
def extract_and_clean_json(text: str) -> str:
    """
    Extracts a JSON string from the AI's potentially verbose output and cleans common errors.
    """
    match = re.search(r'{.*}', text, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in AI response.")
        return "{}"

    json_str = match.group(0)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str) # Remove trailing commas
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    return json_str


# --- 8. API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify model status."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model or Processor not loaded.")
    return {"status": "healthy", "device": device}


@app.post("/query/text")
async def text_query(request: TextQuery):
    """Processes a text-only query and returns a structured JSON analysis."""
    start_time = time.time()
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": request.system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": request.question}]}
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            generation = generation[0][input_len:]
        
        raw_output = processor.decode(generation, skip_special_tokens=True)
        json_str = extract_and_clean_json(raw_output)
        
        # Validate and return JSON
        try:
            json.loads(json_str) # Just to validate
            return Response(content=json_str, media_type="application/json")
        except json.JSONDecodeError as e:
            logger.error(f"AI response was not valid JSON after cleaning: {e}\nRaw Output: {raw_output}")
            error_payload = {"error": "AI did not produce valid JSON.", "raw_output": raw_output}
            return JSONResponse(status_code=500, content=error_payload)

    except Exception as e:
        logger.error(f"Error processing text query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form("Describe this medical image in detail"),
    system_prompt: str = Form(MEDICAL_IMAGE_SYSTEM_PROMPT)
):
    """Processes an image query and returns a structured JSON analysis."""
    start_time = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": image}]}
        ]
        
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            generation = generation[0][input_len:]
            
        raw_output = processor.decode(generation, skip_special_tokens=True)
        json_str = extract_and_clean_json(raw_output)

        try:
            json.loads(json_str) # Validate
            return Response(content=json_str, media_type="application/json")
        except json.JSONDecodeError as e:
            logger.error(f"AI image response not valid JSON: {e}\nRaw Output: {raw_output}")
            error_payload = {"error": "AI did not produce valid JSON.", "raw_output": raw_output}
            return JSONResponse(status_code=500, content=error_payload)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/medical_transcript")
async def process_medical_transcript(
    medical_transcript: str = Form(...),
    system_prompt: str = Form(SOAP_NOTE_SYSTEM_PROMPT),
    file: Optional[UploadFile] = File(None)
):
    """Processes a medical transcript (and optional image) into a SOAP note."""
    start_time = time.time()
    try:
        user_content = [{"type": "text", "text": medical_transcript}]
        image_data = None
        
        if file:
            contents = await file.read()
            image_data = Image.open(io.BytesIO(contents)).convert("RGB")
            user_content.append({"type": "image", "image": image_data})
            logger.info(f"Image '{file.filename}' processed for transcript.")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content}
        ]
        
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            generation = generation[0][input_len:]
            
        soap_note = processor.decode(generation, skip_special_tokens=True)
        
        return JSONResponse({
            "soap_note": soap_note,
            "processing_time_seconds": time.time() - start_time,
        })

    except Exception as e:
        logger.error(f"Error processing medical transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 9. Main Execution Block ---
if __name__ == "__main__":
    """
    This block allows the script to be run directly.
    Example: python medgemma_api_server.py
    """
    port = 8000
    logger.info(f"Starting Uvicorn server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
