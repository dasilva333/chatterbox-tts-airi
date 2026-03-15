import torch
import torch.nn.functional as F
import numpy as np
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any
import uvicorn
import os
import asyncio
import re
import json
import random
import argparse
from pathlib import Path
import time

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
PRESETS_PATH = BASE_DIR / "presets.json"

profiles = {}
presets = {}
LAST_PROFILES_MTIME = 0
LAST_PRESETS_MTIME = 0

VOICE_FILE_EXTENSIONS = [".wav", ".mp3", ".ogg"]

def load_config():
    global profiles, presets, LAST_PROFILES_MTIME, LAST_PRESETS_MTIME
    
    # Reload Profiles
    if PROFILES_PATH.exists():
        mtime = os.path.getmtime(PROFILES_PATH)
        if mtime > LAST_PROFILES_MTIME:
            try:
                with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                LAST_PROFILES_MTIME = mtime
                print(f"Reloaded profiles.json (updated {time.ctime(mtime)})")
            except Exception as e:
                print(f"Error loading profiles.json: {e}")
    
    # Reload Presets
    if PRESETS_PATH.exists():
        mtime = os.path.getmtime(PRESETS_PATH)
        if mtime > LAST_PRESETS_MTIME:
            try:
                with open(PRESETS_PATH, "r", encoding="utf-8") as f:
                    presets = json.load(f)
                LAST_PRESETS_MTIME = mtime
                print(f"Reloaded presets.json (updated {time.ctime(mtime)})")
            except Exception as e:
                print(f"Error loading presets.json: {e}")

def save_json(path: Path, data: Dict[str, Any]):
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, path)

def persist_profiles():
    global LAST_PROFILES_MTIME
    save_json(PROFILES_PATH, profiles)
    LAST_PROFILES_MTIME = os.path.getmtime(PROFILES_PATH)

def persist_presets():
    global LAST_PRESETS_MTIME
    save_json(PRESETS_PATH, presets)
    LAST_PRESETS_MTIME = os.path.getmtime(PRESETS_PATH)

def ensure_non_empty_id(value: str, kind: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{kind} id is required")
    return normalized

# Initial load
load_config()

ACTIVE_PROFILE_NAME = args.mannerisms
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
IVY_VOICE_PATH = str(BASE_DIR / "voices" / "ivy.mp3")
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

def list_voice_files() -> List[str]:
    voices_dir = BASE_DIR / "voices"
    if not voices_dir.exists():
        return []

    return sorted([
        file.stem.lower()
        for file in voices_dir.iterdir()
        if file.suffix.lower() in VOICE_FILE_EXTENSIONS
    ])

class EmoticonRule(BaseModel):
    pattern: str
    replacement: str

class NarrativeSettings(BaseModel):
    rate: Optional[str] = None
    volume: Optional[str] = None

class PresetPayload(BaseModel):
    id: str
    voice_file: str
    tts_model: str = "full"
    exaggeration: float = 0.0
    mannerism_profile: str = ""
    ui_expressions: List[str] = Field(default_factory=list)
    ui_mannerisms: List[str] = Field(default_factory=list)

class ProfilePayload(BaseModel):
    id: str
    hmph: str = ""
    tilde: List[str] = Field(default_factory=list)
    emoticons: List[EmoticonRule] = Field(default_factory=list)
    narrative: Optional[NarrativeSettings] = None

def normalize_preset(payload: PresetPayload) -> Dict[str, Any]:
    voice_files = list_voice_files()
    voice_file = payload.voice_file.strip().lower()
    if voice_file and voice_files and voice_file not in voice_files:
        raise HTTPException(status_code=400, detail=f"Unknown voice file '{voice_file}'")

    mannerism_profile = payload.mannerism_profile.strip()
    if mannerism_profile and mannerism_profile not in profiles:
        raise HTTPException(status_code=400, detail=f"Unknown profile '{mannerism_profile}'")

    tts_model = payload.tts_model.strip().lower()
    if tts_model not in {"full", "turbo"}:
        raise HTTPException(status_code=400, detail=f"Unknown tts_model '{payload.tts_model}'")

    return {
        "voice_file": voice_file,
        "tts_model": tts_model,
        "exaggeration": float(payload.exaggeration),
        "mannerism_profile": mannerism_profile,
        "ui_expressions": [item.strip() for item in payload.ui_expressions if item.strip()],
        "ui_mannerisms": [item.strip() for item in payload.ui_mannerisms if item.strip()],
    }

def normalize_profile(payload: ProfilePayload) -> Dict[str, Any]:
    emoticons = []
    for rule in payload.emoticons:
        pattern = rule.pattern.strip()
        replacement = rule.replacement.strip()
        if not pattern or not replacement:
            continue
        emoticons.append({
            "pattern": pattern,
            "replacement": replacement,
        })

    profile: Dict[str, Any] = {
        "tilde": [item.strip() for item in payload.tilde if item.strip()],
        "emoticons": emoticons,
    }

    hmph = payload.hmph.strip()
    if hmph:
        profile["hmph"] = hmph

    if payload.narrative:
        narrative = {
            key: value.strip()
            for key, value in payload.narrative.model_dump().items()
            if isinstance(value, str) and value.strip()
        }
        if narrative:
            profile["narrative"] = narrative

    return profile

def presets_using_profile(profile_id: str) -> List[str]:
    return sorted([
        preset_id
        for preset_id, preset_data in presets.items()
        if preset_data.get("mannerism_profile") == profile_id
    ])

class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "ivy"
    response_format: str = "mp3"
    speed: float = 1.0
    exaggeration: float = 0.0

@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    load_config()
    start_time = time.time()
    async with synth_lock:
        try:
            # 1. Preset Resolution Logic
            target_voice = request.voice
            target_profile = ACTIVE_PROFILE_NAME
            target_exaggeration = request.exaggeration

            if target_voice in presets:
                preset = presets[target_voice]
                print(f"Resolving Preset: {target_voice}")
                target_voice = preset.get("voice_file", "ivy")
                target_profile = preset.get("mannerism_profile", target_profile)
                target_exaggeration = preset.get("exaggeration", target_exaggeration)

            # 2. Pre-process text based on resolved profile
            processed_input = preprocess_text(request.input, target_profile)
            
            # 3. Resolve voice path (dynamic with Ivy fallback)
            voice_path = resolve_voice_path(target_voice)
            
            print(f"--- Synthesis Request ---")
            print(f"Original Input: {request.input[:50]}...")
            print(f"Processed Input: {processed_input[:50]}...")
            print(f"Requested Voice: {request.voice} -> {target_voice}")
            print(f"Resolved Profile: {target_profile}")
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
                    exaggeration=target_exaggeration
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
                
            latency = time.time() - start_time
            print(f"--- Request Complete ---")
            print(f"Total Latency: {latency:.3f}s")
            
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
    load_config()
    available_voices = []
    
    # 1. Native Voices
    for voice_file in list_voice_files():
        available_voices.append({
            "voice_id": voice_file,
            "name": voice_file.capitalize(),
            "preview_url": None,
            "provider": "chatterbox",
            "type": "native"
        })
    
    # 2. Virtual Voices (Presets)
    for preset_id, preset_data in presets.items():
        available_voices.append({
            "voice_id": preset_id,
            "name": preset_id,
            "provider": "chatterbox",
            "type": "virtual",
            "metadata": preset_data
        })
    
    return {"voices": available_voices}

@app.get("/chatterbox/capabilities")
async def get_capabilities():
    """Returns available voice files, mannerism profiles, and TTS modes."""
    load_config()

    return {
        "voices": list_voice_files(),
        "profiles": list(profiles.keys()),
        "modes": ["full", "turbo"]
    }

@app.get("/chatterbox/presets")
async def get_presets():
    load_config()
    return {
        "presets": [
            {"id": preset_id, **preset_data}
            for preset_id, preset_data in presets.items()
        ]
    }

@app.post("/chatterbox/presets")
async def create_preset(payload: PresetPayload):
    load_config()
    preset_id = ensure_non_empty_id(payload.id, "Preset")
    if preset_id in presets:
        raise HTTPException(status_code=409, detail=f"Preset '{preset_id}' already exists")

    presets[preset_id] = normalize_preset(payload)
    persist_presets()
    return {"preset": {"id": preset_id, **presets[preset_id]}}

@app.put("/chatterbox/presets/{preset_id}")
async def update_preset(preset_id: str, payload: PresetPayload):
    load_config()
    preset_id = ensure_non_empty_id(preset_id, "Preset")
    body_id = ensure_non_empty_id(payload.id, "Preset")
    if preset_id not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    next_data = normalize_preset(payload)
    if body_id != preset_id:
        if body_id in presets:
            raise HTTPException(status_code=409, detail=f"Preset '{body_id}' already exists")
        del presets[preset_id]
        preset_id = body_id

    presets[preset_id] = next_data
    persist_presets()
    return {"preset": {"id": preset_id, **presets[preset_id]}}

@app.delete("/chatterbox/presets/{preset_id}")
async def delete_preset(preset_id: str):
    load_config()
    preset_id = ensure_non_empty_id(preset_id, "Preset")
    if preset_id not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    deleted = presets.pop(preset_id)
    persist_presets()
    return {"deleted": {"id": preset_id, **deleted}}

@app.get("/chatterbox/profiles")
async def get_profiles():
    load_config()
    return {
        "profiles": [
            {"id": profile_id, **profile_data}
            for profile_id, profile_data in profiles.items()
        ]
    }

@app.post("/chatterbox/profiles")
async def create_profile(payload: ProfilePayload):
    load_config()
    profile_id = ensure_non_empty_id(payload.id, "Profile")
    if profile_id in profiles:
        raise HTTPException(status_code=409, detail=f"Profile '{profile_id}' already exists")

    profiles[profile_id] = normalize_profile(payload)
    persist_profiles()
    return {"profile": {"id": profile_id, **profiles[profile_id]}}

@app.put("/chatterbox/profiles/{profile_id}")
async def update_profile(profile_id: str, payload: ProfilePayload):
    load_config()
    profile_id = ensure_non_empty_id(profile_id, "Profile")
    body_id = ensure_non_empty_id(payload.id, "Profile")
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    next_data = normalize_profile(payload)
    if body_id != profile_id:
        dependent_presets = presets_using_profile(profile_id)
        if dependent_presets:
            raise HTTPException(
                status_code=409,
                detail=f"Profile '{profile_id}' is used by presets: {', '.join(dependent_presets)}"
            )
        if body_id in profiles:
            raise HTTPException(status_code=409, detail=f"Profile '{body_id}' already exists")
        del profiles[profile_id]
        profile_id = body_id

    profiles[profile_id] = next_data
    persist_profiles()
    return {"profile": {"id": profile_id, **profiles[profile_id]}}

@app.delete("/chatterbox/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    load_config()
    profile_id = ensure_non_empty_id(profile_id, "Profile")
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")

    dependent_presets = presets_using_profile(profile_id)
    if dependent_presets:
        raise HTTPException(
            status_code=409,
            detail=f"Profile '{profile_id}' is used by presets: {', '.join(dependent_presets)}"
        )

    deleted = profiles.pop(profile_id)
    persist_profiles()
    return {"deleted": {"id": profile_id, **deleted}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
