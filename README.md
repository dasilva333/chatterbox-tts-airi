# Chatterbox TTS Server

Chatterbox is a high-performance, OpenAI-compatible Text-to-Speech (TTS) server designed for expressive character synthesis. It features dynamic voice cloning, persona-based text preprocessing, and advanced emotion tag support.

## Features
- **OpenAI Compatible**: Seamlessly integrates with clients like SillyTavern.
- **Voice Cloning**: Clone any voice by simply placing a 6-second `.wav` or `.mp3` clip in the `voices/` directory.
- **Mannerisms**: Customize character-specific mannerisms, fillers (e.g., `~` mappings), and text replacements via `profiles.json`.
- **Emotion Tags**: Trigger specific sounds like `[laughter]`, `[sigh]`, or `[whisper]` using square bracket tags.
- **Performance Optimized**: Features asynchronous request queuing and model priming for near-instant synthesis.
- **Turbo Mode**: Supports the high-speed `ChatterboxTurboTTS` for near-real-time synthesis.
- **OGG Opus Support**: Natively streams high-quality, low-bandwidth OGG Opus audio.

---

## Prerequisites (Authentication)
The Chatterbox models (especially the Turbo version) are hosted on Hugging Face. To download them automatically, you must set an environment variable with your Hugging Face Access Token:

1.  **Get a Token**: Create a "Read" token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2.  **Set Environment Variable**:
    - **Windows**: `setx HF_TOKEN "your_token_here"` (restart your terminal)
    - **Linux/bash**: `export HF_TOKEN="your_token_here"`

### One-Line Setup (Fast Track)
If you don't want to set a system-wide variable, you can run this **once** to download the model and start the server:

```batch
set HF_TOKEN=your_token_here && run_server.bat --mannerisms=catgirl --turbo
```
> [!NOTE]  
> You only need the token for the **first run** while the model is downloading. After that, the files are cached locally and you can run `run_server.bat` normally.

---

## Installation & Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dasilva333/chatterbox-tts-airi.git
   cd chatterbox-tts-airi
   ```

2. **Initialize Environment**:
   Run the following to set up the virtual environment and install dependencies:
   ```bash
   py -m venv venv
   .\venv\Scripts\activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install git+https://github.com/resemble-ai/chatterbox.git fastapi uvicorn soundfile pydantic
   ```

3. **Prepare Voices**:
   Place your reference audio clips (6 seconds recommended) in the `voices/` folder.
   - `ivy.mp3` (Default)
   - `zenbara.wav` (New character)

---

## Usage

### Starting the Server
Run the provided batch file to start the FastAPI server:
```bash
run_server.bat --mannerisms=kappybara --turbo
```
- `--mannerisms`: Choose a character mannerism from `profiles.json`.
- `--turbo`: Use the high-speed Turbo model (requires higher VRAM).
- `--port`: Default is `8090`.

### CLI Runner
For quick one-off generation:
```bash
python runner.py zenbara "Hello Phil... [sigh] ~ how are you? ~" --turbo
```

---

## Configuration & Customization

### Mannerisms (`profiles.json`)
Manage character-specific logic:
- **`tilde`**: Mappings for the `~` character (e.g., `nyan` for catgirl, `bro` for kappybara).
- **`hmph`**: Custom pronunciations for "hmph" variants (e.g., `hahmf`).
- **`emoticons`**: Regex-based replacements for patterns like `0_0`.
- **`narrative`**: Character-specific speech settings for `*text*` (rate, volume).

### Emotion & Expressiveness Tips
- **Exact Tags**: Use the tags exactly as written in `supported_tags.md`.
- **Exaggeration**: The default is set to **`0.0`** (natural baseline). Increase this (e.g., to `1.0` or higher) for more intense emotional delivery.

---

## Benchmark Results (CUDA)
The results reflect the true capabilities of Chatterbox when hardware acceleration is properly applied via `torch.cuda`:

| Length | Chars | Time (s) | Chars/s |
| :--- | :--- | :--- | :--- |
| 20 chars | 41 | 7.609 | 5.39 |
| 60 chars | 93 | 13.182 | 7.06 |
| 300 chars | 386 | 81.751 | 4.72 |

---

## Project Structure
- **`server.py`**: The FastAPI wrapper. Scans `voices/`, supports `--profile`, and returns OGG Opus.
- **`runner.py`**: CLI script for direct OGG Opus generation.
- **`benchmark.py`**: Script used for gathering generation timings.
- **`supported_tags.md`**: Reference list of verified sound/emotion tokens.
- **`profiles.json`**: JSON store for character mannerisms.
- **`voices/`**: Directory for voice cloning source files.

---

## Voice Discovery
Clients can discover available voices by calling:
- `GET /v1/voices`
- `GET /v1/audio/voices`

This returns a JSON list of all voice files detected in the `voices/` folder. Any unknown voice requested via API will gracefully fallback to **Ivy**.

---

*nyan! [laughter] ... [sigh] ... chill bro.*
