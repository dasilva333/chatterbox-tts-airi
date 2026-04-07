import argparse
import requests
import re
import os
import subprocess
from pathlib import Path

SERVER_URL = "http://10.0.0.91:8090/v1/audio/speech"

def main():
    parser = argparse.ArgumentParser(description="Chatterbox Dialoger")
    parser.add_argument("input", help="Path to the dialog text file")
    parser.add_argument("--out", default="dialog_output.mp3", help="Final stitched filename")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        return

    # 1. Parse the dialog file
    # Format: [voice_id] text
    segments = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            match = re.match(r"\[([^\]]+)\]\s*(.*)", line)
            if match:
                voice_id = match.group(1).strip()
                text = match.group(2).strip()
                segments.append((voice_id, text))
            else:
                print(f"Skipping malformed line: {line}")

    if not segments:
        print("No valid dialog segments found.")
        return

    # 2. Generate audio for each segment
    temp_files = []
    print(f"Generating {len(segments)} segments...")
    
    for i, (voice_id, text) in enumerate(segments):
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
                temp_file = f"temp_seg_{i}.ogg"
                with open(temp_file, "wb") as f:
                    f.write(response.content)
                temp_files.append(temp_file)
            elif response.status_code == 204:
                print(f"Skipping empty response for segment {i}")
            else:
                print(f"Error generating segment {i}: {response.text}")
        except Exception as e:
            print(f"Connection failed at segment {i}: {e}")
            break

    # 3. Stitch using ffmpeg
    if temp_files:
        print("Stitching segments with ffmpeg...")
        # Create a concat list file
        concat_list = "concat_list.txt"
        with open(concat_list, "w") as f:
            for tf in temp_files:
                f.write(f"file '{tf}'\n")
        
        try:
            # Re-encode to the final format to ensure compatibility (especially for mp3)
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, args.out]
            subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
            print(f"Success! Final dialog saved to: {args.out}")
        except Exception as e:
            print(f"Stitching failed: {e}")
        finally:
            # 4. Cleanup
            if os.path.exists(concat_list):
                os.remove(concat_list)
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)

if __name__ == "__main__":
    main()
