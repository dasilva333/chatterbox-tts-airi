# Chatterbox TTS Server

Chatterbox is a high-performance, OpenAI-compatible Text-to-Speech (TTS) server optimized for **AIRI**. It features dynamic voice cloning, persona-based text preprocessing, and a specialized queuing system to handle high-concurrency requests.

## Features
- **OpenAI Compatible**: Seamlessly integrates with any OpenAI-compatible TTS client.
- **Voice Cloning**: Clone any voice by simply placing a 6-second `.wav` or `.mp3` clip in the `voices/` directory.
- **Mannerisms**: Customize character-specific mannerisms, fillers (e.g., `~` mappings), and text replacements via `profiles.json`.
- **Emotion Tags**: Trigger specific sounds like `[laughter]`, `[sigh]`, or `[whisper]` using square bracket tags.
- **Optimized for AIRI**: Specially designed to handle AIRI's parallel request pattern by converting them into a high-performance sequential queue, protecting GPU resources while maintaining low-latency responses.
- **Model Priming**: Features automatic model priming for near-instant synthesis during active sessions.
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

## Hardware & Compatibility
Chatterbox is designed to be flexible across different GPU generations. Depending on your hardware, you should choose the appropriate installation path:

### ⚙️ Option A: Standard Hardware (RTX 20/30/40-Series)
Most users should use the standard stable drivers:
- **Environment**: CUDA 11.8 or 12.1 (Stable)
- **Performance**: Reliable, standard synthesis speeds.

### 🚀 Option B: Next-Gen Hardware (RTX 50-Series / Blackwell)
If you have an **RTX 5090** or similar and encounter kernel errors (e.g., `torchvision::nms` missing):
- **Environment**: PyTorch Nightly with **cu128** support.
- **Performance**: High-throughput synthesis, especially in **Turbo Mode**.

---

## Installation & Setup

### ⚡ Automated Setup (Recommended)
Run the provided installation script to automatically create the virtual environment and install base dependencies:
```bash
install.bat
```

### 🔧 Manual Installation
1. **Prepare Environment**:
   ```bash
   py -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Hardware-Specific Optimization (Crucial)**:
   - **For Standard GPUs**: (Already handled by `install.bat` / `requirements.txt`)
   - **For RTX 50-Series**: Run these commands inside your activated `venv`:
     ```bash
     pip uninstall torch torchvision torchaudio
     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
     ```

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

### RTX 5090 (Blackwell)
*Environment: PyTorch Nightly cu128*

**Turbo Model (`--turbo`)**
| Length | Chars | Time (s) | Chars/s |
| :--- | :--- | :--- | :--- |
| 20 chars | 41 | 2.515 | 16.30 |
| 60 chars | 93 | 3.849 | 24.16 |
| 300 chars | 386 | 17.953 | 21.50 |

**Standard Model**
| Length | Chars | Time (s) | Chars/s |
| :--- | :--- | :--- | :--- |
| 20 chars | 41 | 7.062 | 5.81 |
| 60 chars | 93 | 8.071 | 11.52 |
| 300 chars | 386 | 32.237 | 11.97 |

---

### RTX 4070 Mobile (Baseline)
*Environment: Standard cu118*

**Turbo Model (`--turbo`)**
| Length | Chars | Time (s) | Chars/s |
| :--- | :--- | :--- | :--- |
| 20 chars | 41 | 6.054 | 6.77 |
| 60 chars | 93 | 7.921 | 11.74 |
| 300 chars | 386 | 16.499 | 23.40 |

**Standard Model**
| Length | Chars | Time (s) | Chars/s |
| :--- | :--- | :--- | :--- |
| 20 chars | 41 | 7.609 | 5.39 |
| 60 chars | 93 | 13.182 | 7.06 |
| 300 chars | 386 | 81.751 | 4.72 |

---

## Project Structure
- **`server.py`**: The FastAPI wrapper. Scans `voices/`, supports `--profile`, and returns OGG Opus.
- **`install.bat`**: Automated setup and dependency installation script.
- **`requirements.txt`**: Pinned dependencies for environment stability.
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
