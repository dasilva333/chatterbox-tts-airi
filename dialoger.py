import argparse
import requests
import re
import os
import subprocess
import random
from pathlib import Path

SERVER_URL = "http://10.0.0.91:8090/v1/audio/speech"

def generate_silence(duration_ms, output_path):
    """Generates a silent ogg file of specified duration."""
    duration_sec = duration_ms / 1000.0
    # Use 48000 Hz to match TTS output and avoid filter mismatch
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono", 
        "-t", str(duration_sec), "-c:a", "libopus", output_path
    ]
    subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)

def main():
    parser = argparse.ArgumentParser(description="Chatterbox Dialoger Pro")
    parser.add_argument("input", help="Path to the dialog text file")
    parser.add_argument("--out", default="dialog_output.mp3", help="Final stitched filename")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        return

    # 1. Parse the dialog file
    segments = []
    panning_overrides = {}
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Look for panning overrides at the top: (pan:voice_id=0.2)
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check for pan overrides
        pan_match = re.match(r"\(pan:([^=]+)=([\d.-]+)\)", line)
        if pan_match:
            voice_id = pan_match.group(1).strip()
            pan_val = float(pan_match.group(2))
            panning_overrides[voice_id] = pan_val
            continue
            
        # Check for segments
        segment_match = re.match(r"\[([^\]]+)\]\s*(.*)", line)
        if segment_match:
            voice_id = segment_match.group(1).strip()
            text = segment_match.group(2).strip()
            
            # Handle [pause:ms] syntax
            if ":" in voice_id and voice_id.startswith("pause"):
                try:
                    ms_val = int(voice_id.split(":")[1])
                    segments.append(("pause", str(ms_val)))
                    continue
                except ValueError:
                    pass
            
            segments.append((voice_id, text))
        else:
            print(f"Skipping malformed line: {line}")

    if not segments:
        print("No valid dialog segments found.")
        return

    # --- Validation Phase ---
    voices_dir = Path("voices")
    voice_files = set()
    for ext in [".mp3", ".ogg", ".wav", ".m4a"]:
        for f in voices_dir.glob(f"*{ext}"):
            voice_files.add(f.stem.lower())
    
    unique_requested_voices = set(v for v, t in segments if v != "pause")
    missing_voices = []
    for v in unique_requested_voices:
        if v.lower() not in voice_files:
            missing_voices.append(v)
            
    if missing_voices:
        print("\n❌ VALIDATION FAILED: Script uses voices not found in 'voices/' directory.")
        print(f"Missing Voice IDs: {', '.join(missing_voices)}")
        print("\nPlease fix the script or the voice files before proceeding.")
        return
    # --- End Validation Phase ---

    # Assign random panning for any voice not overridden (excluding silence)
    voice_panning = panning_overrides.copy()
    unique_voices = set(v for v, t in segments if v != "pause")
    for v in unique_voices:
        if v not in voice_panning:
            voice_panning[v] = round(random.uniform(-0.4, 0.4), 2)
            print(f"Assigned spatial position for {v}: {voice_panning[v]}")

    # 2. Generate audio for each segment
    temp_files = []
    print(f"Generating {len(segments)} segments...")
    
    for i, (voice_id, text) in enumerate(segments):
        temp_file = f"temp_seg_{i}.ogg"
        
        if voice_id == "pause":
            try:
                ms = int(text) if text else 500
                print(f"[{i+1}/{len(segments)}] [SILENCE] {ms}ms")
                generate_silence(ms, temp_file)
                temp_files.append((temp_file, 0)) # No panning for silence
            except ValueError:
                print(f"Invalid pause duration: {text}")
            continue

        print(f"[{i+1}/{len(segments)}] [{voice_id}] {text[:30]}...")
        
        payload = {
            "model": "chatterbox",
            "input": text,
            "voice": voice_id,
            "response_format": "ogg"
        }
        
        try:
            response = requests.post(SERVER_URL, json=payload)
            if response.status_code == 200:
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                temp_files.append((temp_file, voice_panning.get(voice_id, 0)))
            elif response.status_code == 204:
                print(f"Skipping empty response for segment {i}")
            else:
                print(f"Error generating segment {i}: {response.text}")
        except Exception as e:
            print(f"Connection failed at segment {i}: {e}")
            break

    # 3. Stitch with Normalization and Panning
    if temp_files:
        print("Stitching segments with normalization and panning...")
        
        # Build complex filter string
        filter_complex = ""
        input_args = []
        for i, (f, pan) in enumerate(temp_files):
            input_args.extend(["-i", f])
            left = 0.5 * (1.0 - pan)
            right = 0.5 * (1.0 + pan)
            filter_complex += f"[{i}:a]aresample=48000,pan=stereo|c0={left:.2f}*c0|c1={right:.2f}*c0[a{i}];"
        
        filter_complex += "".join([f"[a{i}]" for i in range(len(temp_files))])
        filter_complex += f"concat=n={len(temp_files)}:v=0:a=1[outa];"
        filter_complex += "[outa]loudnorm[finala]"
        
        try:
            cmd = ["ffmpeg", "-y"] + input_args + ["-filter_complex", filter_complex, "-map", "[finala]", args.out]
            subprocess.run(cmd, check=True)
            print(f"Success! Levelled-up dialog saved to: {args.out}")
        except Exception as e:
            print(f"Stitching failed: {e}")
        finally:
            # 4. Cleanup
            for tf, _ in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)

if __name__ == "__main__":
    main()
