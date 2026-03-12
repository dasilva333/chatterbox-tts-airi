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
parser.add_argument("--mannerisms", default="catgirl", help="Starting mannerisms from profiles.json (default: catgirl)")
parser.add_argument("--turbo", action="store_true", help="Use the high-speed Turbo model")
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

ACTIVE_PROFILE_NAME = args.mannerisms
ACTIVE_PROFILE = profiles.get(ACTIVE_PROFILE_NAME, {})
print(f"Active Mannerisms: {ACTIVE_PROFILE_NAME}")

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

    # 5. Narrative (*text*) and Mutters ((text)) -> [whisper] (prefix only)
    text = re.sub(r"\*([^*]+)\*", r" [whisper] \1 ", text)
    text = re.sub(r"(\([^)]+\)|\[[^\]]+\])", r" [whisper] \1 ", text)

    # 6. Dramatic Ellipsis -> [sigh]
    text = text.replace("...", " [sigh] ")

    # 7. Strip Emojis (similar to aws-polly)
    text = EMOJI_REGEX.sub(" ", text)

    # Final cleanup of multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Initialize the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if args.turbo:
    if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN environment variable not found. Downloading the Turbo model may fail if you are not authenticated.")
        print("Please set your token: 'set HF_TOKEN=your_token_here' before running.")
    
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    print(f"Loading Chatterbox Turbo model on {DEVICE}...")
    model = ChatterboxTurboTTS.from_pretrained(DEVICE)
else:
    from chatterbox import ChatterboxTTS
    print(f"Loading Chatterbox model on {DEVICE}...")
    model = ChatterboxTTS.from_pretrained(DEVICE)
print("Model loaded successfully.")

# Global lock for sequential processing
synth_lock = asyncio.Lock()

# Hardcode Ivy voice path and prime the model
IVY_VOICE_PATH = str(BASE_DIR / "voices" / "zenbara.wav")
print(f"Pre-sampling/Priming default voice: {IVY_VOICE_PATH}")
# This calculates the embeddings and keeps them in model.conds
model.prepare_conditionals(IVY_VOICE_PATH)
print("Priming complete. Model is warm.")

# Cache for primed voices to avoid redundant prepare_conditionals calls
PRIMED_VOICES = { "ivy": IVY_VOICE_PATH }

def resolve_voice_path(requested_voice: str) -> str:
    """Find voice in voices/ folder or fallback to ivy."""
    voices_dir = BASE_DIR / "voices"
    requested_voice = requested_voice.lower().strip()
    
    # Check common extensions
    for ext in [".wav", ".mp3", ".ogg"]:
        path = voices_dir / f"{requested_voice}{ext}"
        if path.exists():
            return str(path)
            
    # Fallback to ivy
    print(f"Warning: Voice '{requested_voice}' not found in {voices_dir}. Falling back to Ivy.")
    return IVY_VOICE_PATH

class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "ivy"
    response_format: str = "mp3"
    speed: float = 1.0
    exaggeration: float = 0.0

@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    async with synth_lock:
        try:
            # Pre-process text based on the active profile
            processed_input = preprocess_text(request.input, ACTIVE_PROFILE_NAME)
            
            # Resolve voice path (dynamic with Ivy fallback)
            voice_path = resolve_voice_path(request.voice)
            
            print(f"--- Synthesis Request ---")
            print(f"Original Input: {request.input[:50]}...")
            print(f"Processed Input: {processed_input[:50]}...")
            print(f"Requested Voice: {request.voice}")
            print(f"Resolved Voice Path: {voice_path}")

            # Dynamic Priming: If we haven't seen this voice before, prime it
            voice_name = Path(voice_path).stem.lower()
            if voice_name not in PRIMED_VOICES:
                print(f"Priming new voice on the fly: {voice_name}")
                model.prepare_conditionals(voice_path)
                PRIMED_VOICES[voice_name] = voice_path

            # Generate audio
            if args.turbo:
                # Turbo model generate doesn't take exaggeration
                wav_tensor = model.generate(
                    processed_input, 
                    audio_prompt_path=voice_path
                )
            else:
                wav_tensor = model.generate(
                    processed_input, 
                    audio_prompt_path=voice_path, 
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

@app.get("/v1/voices")
@app.get("/v1/audio/voices")
async def list_voices():
    voices_dir = BASE_DIR / "voices"
    available_voices = []
    
    if voices_dir.exists():
        for file in voices_dir.iterdir():
            if file.suffix.lower() in [".wav", ".mp3", ".ogg"]:
                available_voices.append({
                    "voice_id": file.stem.lower(),
                    "name": file.stem.capitalize(),
                    "preview_url": None,
                    "provider": "chatterbox"
                })
    
    return {"voices": available_voices}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
