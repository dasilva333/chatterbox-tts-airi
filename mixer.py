import argparse
import subprocess
import os
from pathlib import Path

def get_duration(file_path):
    """Gets duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def main():
    parser = argparse.ArgumentParser(description="Chatterbox Audio Mixer Pro")
    parser.add_argument("--dialog", required=True, help="Path to the dialog MP3")
    parser.add_argument("--bg", required=True, help="Path to the background track MP3")
    parser.add_argument("--out", default="final_mix.mp3", help="Output filename")
    parser.add_argument("--dialog-vol", type=float, default=1.0, help="Volume for dialog (0.0 to 2.0)")
    parser.add_argument("--bg-vol", type=float, default=0.15, help="Volume for background (0.0 to 1.0)")
    
    args = parser.parse_args()

    dialog_path = Path(args.dialog)
    bg_path = Path(args.bg)

    if not dialog_path.exists() or not bg_path.exists():
        print("Error: Dialog or Background file not found.")
        return

    print(f"🎚️ Mixing: {dialog_path.name} + {bg_path.name}")
    print(f"🛠️ Settings: Dialog Vol={args.dialog_vol}, BG Vol={args.bg_vol}")

    # Get dialog duration for absolute truncation
    try:
        duration = get_duration(dialog_path)
        print(f"⏱️ Target Duration: {duration:.2f}s")
    except Exception as e:
        print(f"Warning: Could not determine duration, using amix duration policy. ({e})")
        duration = None

    # [1:a]aloop=loop=-1:size=2e9[bgloop] -> loop indefinitely
    # [bgloop]volume={args.bg_vol}[bg_ready]
    # [0:a]volume={args.dialog_vol}[dialog_ready]
    # [dialog_ready][bg_ready]amix=inputs=2:dropout_transition=0[outa]
    
    filter_complex = (
        f"[1:a]aloop=loop=-1:size=2e9,volume={args.bg_vol}[bg_ready];"
        f"[0:a]volume={args.dialog_vol}[dialog_ready];"
        f"[dialog_ready][bg_ready]amix=inputs=2:dropout_transition=0[outa]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(dialog_path),
        "-i", str(bg_path),
        "-filter_complex", filter_complex,
        "-map", "[outa]"
    ]

    # Explicitly truncate if we have a duration
    if duration:
        cmd.extend(["-t", str(duration)])

    cmd.append(args.out)

    try:
        print("Mixing with ffmpeg...")
        subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)
        print(f"✨ Success! Final mix saved to: {args.out}")
    except Exception as e:
        print(f"Mixing failed: {e}")

if __name__ == "__main__":
    main()
