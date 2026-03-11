import sys
import torch
import soundfile as sf
from chatterbox import ChatterboxTTS
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chatterbox TTS Runner")
    parser.add_argument("voice", help="Voice name (link to voices/<name>.mp3 or .wav)")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--turbo", action="store_true", help="Use the Turbo model")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0 to 1.0+)")
    
    args = parser.parse_args()

    voice_name = args.voice
    text = args.text
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Chatterbox {'Turbo ' if args.turbo else ''}model on {device}...")
    
    if args.turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device)
    else:
        from chatterbox import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device)

    # Resolve voice path
    voice_path = None
    if voice_name != "default":
        possible_paths = [
            Path("voices") / f"{voice_name}.mp3",
            Path("voices") / f"{voice_name}.wav"
        ]
        for p in possible_paths:
            if p.exists():
                voice_path = str(p)
                break
        
        if not voice_path:
            print(f"Warning: Voice '{voice_name}' not found in 'voices' directory. Using default.")
        else:
            print(f"Using voice condition: {voice_path}")

    print(f"Generating audio for text: \"{text}\" (exaggeration={args.exaggeration})")
    # Generate
    with torch.no_grad():
        if args.turbo:
            # Turbo generate has slightly different parameters
            wav_tensor = model.generate(text, audio_prompt_path=voice_path)
        else:
            wav_tensor = model.generate(text, audio_prompt_path=voice_path, exaggeration=args.exaggeration)
    
    wav_data = wav_tensor.squeeze(0).cpu().numpy()
    
    output_prefix = f"{voice_name}_turbo" if args.turbo else voice_name
    output_file = f"{output_prefix}.ogg"
    print(f"Saving to {output_file}...")
    
    # Try writing as OGG Opus, fallback to OGG Vorbis
    try:
        sf.write(output_file, wav_data, model.sr, format='OGG', subtype='OPUS')
    except Exception as e:
        print(f"Opus not supported natively by soundfile ({e}), falling back to Vorbis...")
        sf.write(output_file, wav_data, model.sr, format='OGG', subtype='VORBIS')

    print("Done!")

if __name__ == "__main__":
    main()
