import os
import requests
import json
from pathlib import Path

VOICES_DIR = Path("/mnt/c/Users/h4rdc/Documents/Github/coding-agent/chatterbox/voices")
WHISPER_URL = "http://localhost:8095/v1/audio/transcriptions"
OUTPUT_FILE = Path("/mnt/c/Users/h4rdc/Documents/Github/coding-agent/chatterbox/voice_vocabulary.json")

# Core family members to transcribe
TARGET_VOICES = [
    "tia-quina",
    "tio-jose",
    "tio-juaco",
    "johnny_2026",
    "cousin-oswal",
    "cousin-pequin",
    "cousin-carolina",
    "cousin-sharon",
    "isabel-dalilas",
    "nayibis",
    "tia-aracelis"
]

def main():
    results = {}
    
    # Load existing results if they exist
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    for voice_id in TARGET_VOICES:
        # Find the file (mp3, ogg, or wav)
        audio_path = None
        for ext in [".mp3", ".ogg", ".wav"]:
            p = VOICES_DIR / f"{voice_id}{ext}"
            if p.exists():
                audio_path = p
                break
        
        if not audio_path:
            print(f"Skipping {voice_id}: No audio file found.")
            continue
            
        print(f"Transcribing {voice_id} ({audio_path.name})...")
        
        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, f"audio/{audio_path.suffix[1:]}")}
                data = {"model": "medium", "language": "es"}
                response = requests.post(WHISPER_URL, files=files, data=data)
                
            if response.status_code == 200:
                text = response.json().get("text", "").strip()
                results[voice_id] = text
                print(f"  Result: {text[:50]}...")
            else:
                print(f"  Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"  Failed to transcribe {voice_id}: {e}")

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Vocabulary saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
