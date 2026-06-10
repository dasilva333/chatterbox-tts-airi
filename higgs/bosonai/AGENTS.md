# AGENTS.md — Higgs Audio v3 TTS (4B)

> Operational guide for AI coding agents. This file is **self-contained**: you can act on it
> even before cloning this repo. For model background, benchmarks, the full language list, and
> citation, see the **[model card README](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/README.md)** — don't duplicate that narrative here.

Higgs Audio v3 TTS is a **4B-parameter, conversational text-to-speech model**: expressive,
low-latency, 100+ languages, zero-shot voice cloning, and inline control over emotion / prosody /
pauses / sound effects mid-utterance.

---

## Step 0 — Pick the right path (read this first)

Choose by constraint, not by habit:

| Goal | Use | Entry point |
|------|-----|-------------|
| Just hear it / try preset voices & avatars | **Live Demo** | https://boson.ai/workspace/avatar |
| Integrate quickly, no GPU, your own voice | **Hosted API** | https://docs.boson.ai/models/higgs-audio-tts/overview |
| Data privacy, custom testing, full control | **Self-host (SGLang-Omni)** | https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/ |
| Inspect weights / config / tokenizer | **Model card (this repo)** | https://huggingface.co/bosonai/higgs-audio-v3-tts-4b |

Deep dive on everything: **Technical blog** → https://boson.ai/blog/higgs-audio-v3-tts

---

## Path A — Hosted API (fastest, no GPU)

> **Authoritative docs:** https://docs.boson.ai/models/higgs-audio-tts/overview
> Get an API key, full field reference, and Python/TypeScript SDK examples there.
> An agent cannot invent a key — if `BOSON_API_KEY` is unset, stop and point the user to this page.

```bash
export BOSON_API_KEY=bai-xxxx          # obtain from https://docs.boson.ai (key format: bai-...)
```

Basic synthesis:

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"higgs-audio-v3-tts","input":"Hello, this is a test."}' \
  --output out.mp3
```

Request fields:

| Field | Notes |
|-------|-------|
| `model` | `"higgs-audio-v3-tts"` |
| `input` | text to synthesize (**required**) |
| `voice` | preset speaker, e.g. `"jake"` |
| `ref_audio` + `ref_text` | URL/base64 clip + its transcript → **voice cloning** |
| `response_format` | `"mp3"` (default) or `"pcm"` (use `pcm` for low-latency streaming) |
| `stream` | `true` for SSE streaming |

> Verify exact field names/limits against the API docs before shipping — the hosted API evolves
> independently of these weights.

---

## Path B — Self-host with SGLang-Omni

### B0 — Preflight: confirm hardware first (do this before pulling anything)

Performance numbers are benchmarked on **1× H100 (80 GB)**. The model is also **confirmed to run on
1× A100 40 GB** — so **~40 GB VRAM is a known-good floor**. Smaller GPUs are **untested** (no data,
not "won't work"). Before deploying:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # GPU present? how much VRAM?
docker --version && docker info | grep -i runtime                      # Docker + NVIDIA runtime ready?
df -h .                                                                 # disk for the ~4B weights + image
```

Rules for the agent:
- **No NVIDIA GPU** → stop. Self-host is not viable; steer the user to **Path A (hosted API)**.
- **≥ 40 GB VRAM (e.g. A100 40 GB, H100)** → known-good; proceed.
- **24 GB (e.g. RTX 4090)** → *reported* to work, **not officially verified**. The ~4B weights fit,
  but expect to lower concurrency / `max_new_tokens` and watch for OOM at the `serve` step.
- **< 24 GB VRAM** → untested. It *may* still run (4B model), but no one has verified it. Warn the
  user, and be ready to lower concurrency / `max_new_tokens` if you hit OOM at the `serve` step.
- **Don't assume** a VRAM number — confirm against the SGLang-Omni cookbook / blog before promising
  a given GPU will work: https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/

### B1 — Install & serve

```bash
# 1. Container
docker pull lmsysorg/sglang-omni:dev
docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
  lmsysorg/sglang-omni:dev /bin/zsh

# 2. Engine
git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v -e .

# 3. Weights
hf download bosonai/higgs-audio-v3-tts-4b

# 4. Serve (OpenAI-compatible audio endpoint)
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

Call the local server:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?"}' \
  --output output.wav
```

**Recommended sampling (voice cloning):** `temperature: 0.8`, `top_k: 50`, `max_new_tokens: 1024`.

Cookbook reference: https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html

---

## Control tags — how to write target text

Embed tags directly in the `input` text to steer emotion, prosody, style, and sound effects.
Format is always `<|category:tag|>`, with two placements:

- **Sentence-level** (emotion / style / prosody speed·pitch·expressive) → put at the sentence start.
- **Inline** (sfx, and prosody `pause` / `long_pause`) → insert at the exact spot in the sentence.
- **`sfx` gotcha:** `<|sfx:cough|>Ahem, ...` — tag first, onomatopoeia attached, **no space**.

```
<|emotion:elation|>Welcome aboard, we are thrilled to have you here!
<|emotion:elation|><|sfx:laughter|>Haha, welcome, we're so happy you're here!
Hello there <|prosody:pause|> and welcome to the show.
```

> **Full 43-tag catalog + rules + examples → [PROMPTING.md](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/PROMPTING.md).**
> Only recognized tags work — anything else degrades output or gets read literally.

For chat formatting, use **[`chat_template.jinja`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/chat_template.jinja)** from the model repo (and the API docs);
**do not hand-assemble the chat prompt** — go through the template.

## Language codes

Only the ISO codes listed in README's supported-languages section are reliable. Codes outside that
list fall back / degrade. → see the [model card README](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/README.md) (`## Supported Languages`).

---

## Repo contents (what's actually here)

This repo (`https://huggingface.co/bosonai/higgs-audio-v3-tts-4b`) is **weights + config**, not an
inference codebase:

- `config.json`, `model.safetensors(.index.json)` — model weights & shape
- `chat_template.jinja` — **authoritative** prompt/chat formatting; respect it
- `tokenizer.json`, `tokenizer_config.json` — tokenizer
- `README.md` — HuggingFace model card (capabilities, benchmarks, languages, citation)
- `LICENSE` — see red line below

## Do / Don't

- ✅ Use `chat_template.jinja` for prompt construction; use the OpenAI-compatible `/v1/audio/speech` shape.
- ✅ Use `pcm` + `stream` for real-time / conversational latency.
- ❌ **Don't use commercially.** License is research & non-commercial
  (`boson-higgs-audio-v3-research-and-non-commercial-license`) — see [LICENSE](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/LICENSE).
- ❌ Don't hardcode the 100-language claim as "any code works" — validate against the supported list.

## Pointers (don't duplicate — link)

All on the model card: `https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/README.md`

- Benchmarks / WER-CER tables → README `## Evaluation Benchmarks`
- Full language list → README `## Supported Languages`
- Control-token catalog → README `## Control Tokens`
- Citation → README `## Citation`
