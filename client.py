import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Chatterbox API Client")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="ivy", help="Voice name or preset ID")
    parser.add_argument("--model", default="chatterbox", help="Model name (OpenAI compatible)")
    parser.add_argument("--format", default="ogg", help="Response format (mp3, ogg, wav)")
    parser.add_argument("--url", default="http://10.0.0.91:8090/v1/audio/speech", help="API URL")
    parser.add_argument("--out", default="output.mp3", help="Output filename")

    args = parser.parse_args()

    payload = {
        "model": args.model,
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format
    }

    print(f"Connecting to {args.url}...")
    try:
        response = requests.post(args.url, json=payload)
        
        if response.status_code == 200:
            # Update output extension if it doesn't match the format
            out_file = args.out
            if not out_file.endswith(f".{args.format}"):
                if "." in out_file:
                    out_file = out_file.rsplit(".", 1)[0] + f".{args.format}"
                else:
                    out_file = out_file + f".{args.format}"
            
            with open(out_file, "wb") as f:
                f.write(response.content)
            print(f"Success! Audio saved to: {out_file}")
        elif response.status_code == 204:
            print("Server returned No Content (likely non-alphanumeric input).")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
