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
import re
import json
import random
import argparse
from pathlib import Path

app = FastAPI(title="Chatterbox OpenAI Compatible Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Chatterbox TTS Server")
parser.add_argument("--profile", default="catgirl", help="Persona profile to use (from profiles.json)")
parser.add_argument("--port", type=int, default=8090, help="Port to run the server on")
args, unknown = parser.parse_known_args()

# Load profiles
BASE_DIR = Path(__file__).parent.absolute()
PROFILES_PATH = BASE_DIR / "profiles.json"
profiles = {}
if PROFILES_PATH.exists():
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
else:
    print(f"Warning: profiles.json not found at {PROFILES_PATH}")

ACTIVE_PROFILE_NAME = args.profile
ACTIVE_PROFILE = profiles.get(ACTIVE_PROFILE_NAME, {})
print(f"Active Profile: {ACTIVE_PROFILE_NAME}")

# Emoji regex: Strips pictographic emoji but preserves digits 0-9.
EMOJI_REGEX = re.compile(r"(?!\d)[\U0001F000-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]\uFE0F?", re.UNICODE)

def preprocess_text(text: str, profile_name: str) -> str:
    profile = profiles.get(profile_name, {})
    if not profile:
        return text

    # 1. Global acronym replacements (case-sensitive)
    text = re.sub(r"\bHR\b", "H.R.", text)
    text = re.sub(r"\bIT\b", "I.T.", text)

    # 2. Hmph variants (complex regex from aws-polly)
    # matches: mph, mmph, humf, hmph, humph, hmpf, humpf, huhmp, hoomp, etc.
    hmph_regex = r"\b(m+p+h+|h+u*m+p*[hf]+|h+u+h+m+p+|h+o+m+p+)\b"
    if "hmph" in profile:
        text = re.sub(hmph_regex, profile["hmph"], text, flags=re.IGNORECASE)

    # 3. Handle profile-specific emoticon replacements
    for emo in profile.get("emoticons", []):
        pattern = emo.get("pattern")
        replacement = emo.get("replacement", "")
        if pattern:
            text = re.sub(pattern, replacement, text)

    # 4. Handle tildes (replace with random filler from the list)
    tilde_fillers = profile.get("tilde", [])
    if tilde_fillers:
        while "~" in text:
            filler = random.choice(tilde_fillers)
            text = text.replace("~", f" {filler} ", 1)

    # 5. Narrative (*text*) -> [whisper] for Chatterbox? 
    # Or just keep as is. Chatterbox doesn't have a direct 'narrative' rate tag yet.
    # But we can wrap it in [whisper] if it exists.
    # text = re.sub(r"\*([^*]+)\*", r" [whisper] \1 [whisper] ", text)

    # 6. Dramatic Ellipsis -> [sigh] or just leave for natural pause
    # text = text.replace("...", " [sigh] ")

    # 7. Strip Emojis (similar to aws-polly)
    text = EMOJI_REGEX.sub(" ", text)

    # Final cleanup of multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Initialize the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Chatterbox model on {DEVICE}...")
model = ChatterboxTTS.from_pretrained(DEVICE)
print("Model loaded successfully.")

# Global lock for sequential processing
synth_lock = asyncio.Lock()

# Hardcode Ivy voice path and prime the model
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
            # Pre-process text based on the active profile
            processed_input = preprocess_text(request.input, ACTIVE_PROFILE_NAME)
            
            print(f"--- Synthesis Request ---")
            print(f"Original Input: {request.input[:50]}...")
            print(f"Processed Input: {processed_input[:50]}...")
            print(f"Request Voice (Ignored): {request.voice}")
            print(f"Using Enforced Voice: {IVY_VOICE_PATH}")

            # Enforce Ivy voice and request-specified (or default) exaggeration
            # Since we primed the model, model.generate will skip sampling if we pass the same path
            wav_tensor = model.generate(
                processed_input, 
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
    uvicorn.run(app, host="0.0.0.0", port=args.port)
