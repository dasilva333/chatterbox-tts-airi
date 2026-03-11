import torch
import torch.nn.functional as F
import numpy as np
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List, Union
from chatterbox import ChatterboxTTS
import uvicorn
import os
import asyncio
from pathlib import Path

app = FastAPI(title="Chatterbox OpenAI Compatible Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox model on {DEVICE}...")
model = ChatterboxTTS.from_pretrained(DEVICE)
print("Model loaded successfully.")

# Global lock for sequential processing
synth_lock = asyncio.Lock()

# Hardcode Ivy voice path and prime the model
BASE_DIR = Path(__file__).parent.absolute()
IVY_VOICE_PATH = str(BASE_DIR / "voices" / "ivy.mp3")
print(f"Pre-sampling/Priming voice: {IVY_VOICE_PATH}")
# This calculates the embeddings and keeps them in model.conds
model.prepare_conditionals(IVY_VOICE_PATH)
print("Priming complete. Model is warm.")

class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "ivy"
    response_format: str = "mp3"
    speed: float = 1.0
    exaggeration: float = 1.0

@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    async with synth_lock:
        try:
            print(f"--- Synthesis Request ---")
            print(f"Input: {request.input[:50]}...")
            print(f"Request Voice (Ignored): {request.voice}")
            print(f"Using Enforced Voice: {IVY_VOICE_PATH}")

            # Enforce Ivy voice and request-specified (or default) exaggeration
            # Since we primed the model, model.generate will skip sampling if we pass the same path
            wav_tensor = model.generate(
                request.input, 
                audio_prompt_path=IVY_VOICE_PATH, 
                exaggeration=request.exaggeration
            )
            
            # Convert to numpy
            wav_data = wav_tensor.squeeze(0).cpu().numpy()
            
            buffer = io.BytesIO()
            if request.response_format in ["opus", "ogg", "mp3"]:
                # mp3 requested (or opus/ogg), try Opus encoding via soundfile
                try:
                    sf.write(buffer, wav_data, model.sr, format='OGG', subtype='OPUS')
                    media_type = "audio/ogg"
                except Exception as e:
                    print(f"Failed to encode as Opus, falling back to Vorbis: {e}")
                    sf.write(buffer, wav_data, model.sr, format='OGG', subtype='VORBIS')
                    media_type = "audio/ogg"
            else:
                sf.write(buffer, wav_data, model.sr, format='WAV')
                media_type = "audio/wav"
                
            return Response(content=buffer.getvalue(), media_type=media_type)
            
        except Exception as e:
            print(f"Error during synthesis: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "chatterbox",
                "object": "model",
                "created": 1700000000,
                "owned_by": "resemble-ai"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
