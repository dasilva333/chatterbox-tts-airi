import gradio as gr


HEAD = r"""
<script>
window.__higgsState = window.__higgsState || {
  repoId: "Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX",
  repo: "https://huggingface.co/Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX/resolve/main/",
  sampleRate: 24000,
  numCodebooks: 8,
  codebookSize: 1026,
  codecCodebookSize: 1024,
  BOC: 1024,
  EOC: 1025,
  ttsId: 151667n,
  audioId: 151670n,
  textId: 151672n,
  tokenizer: null,
  prefillSession: null,
  decodeSession: null,
  vocoderSession: null,
  vocoderArtifact: null,
  transformers: null,
  ortReady: false,
  logs: []
};

window.__higgsLog = function(message, reset=false) {
  const state = window.__higgsState;
  if (reset) state.logs = [];
  state.logs.push(message);
  const rendered = window.__higgsRenderStatus();
  const el = document.getElementById("higgs-live-status");
  if (el) el.innerHTML = rendered;
  return rendered;
};

window.__higgsEscape = function(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
};

window.__higgsRenderStatus = function() {
  return "<pre id=\"higgs-live-status\">" +
    window.__higgsEscape(window.__higgsState.logs.join("\n")) +
    "</pre>";
};

window.__higgsLoadScript = function(src) {
  return new Promise((resolve, reject) => {
    const existing = [...document.scripts].find((s) => s.src === src);
    if (existing && window.ort) return resolve();
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load " + src));
    document.head.appendChild(script);
  });
};

window.__higgsEnsureOrt = async function() {
  const state = window.__higgsState;
  if (state.ortReady && window.ort) return;
  await window.__higgsLoadScript("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.min.js");
  if (!window.ort) throw new Error("ONNX Runtime Web did not initialize.");
  state.ortReady = true;
};

window.__higgsEnsureTokenizer = async function() {
  const state = window.__higgsState;
  if (state.tokenizer) return state.tokenizer;
  if (!state.transformers) {
    state.transformers = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1");
    state.transformers.env.allowLocalModels = false;
  }
  state.tokenizer = await state.transformers.AutoTokenizer.from_pretrained(state.repoId);
  return state.tokenizer;
};

window.__higgsFetchBytes = async function(path) {
  window.__higgsLog("Fetching " + path + "...");
  const response = await fetch(window.__higgsState.repo + path);
  if (!response.ok) throw new Error(path + " fetch failed: " + response.status);
  const bytes = new Uint8Array(await response.arrayBuffer());
  window.__higgsLog("Fetched " + path + " (" + (bytes.length / 1e9).toFixed(2) + " GB).");
  return bytes;
};

window.__higgsLoadOnnx = async function(path) {
  await window.__higgsEnsureOrt();
  const dataPath = path + ".data";
  const modelBytes = await window.__higgsFetchBytes(path);
  const dataBytes = await window.__higgsFetchBytes(dataPath);
  return await window.ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    externalData: [{path: dataPath.split("/").pop(), data: dataBytes}],
    graphOptimizationLevel: "all"
  });
};

window.__higgsLoadAr = async function() {
  const state = window.__higgsState;
  let status = window.__higgsLog("Loading tokenizer for AR generation...", true);
  await window.__higgsEnsureTokenizer();
  status = window.__higgsLog("Tokenizer loaded.");
  status = window.__higgsLog("AR graphs load sequentially during Generate to keep browser memory down.");
  return [window.__higgsRenderStatus(), ""];
};

window.__higgsReleaseSession = async function(name) {
  const state = window.__higgsState;
  const session = state[name];
  state[name] = null;
  if (session && typeof session.release === "function") {
    await session.release();
  }
};

window.__higgsLoadVocoder = async function(artifact) {
  const state = window.__higgsState;
  if (state.vocoderSession && state.vocoderArtifact === artifact) {
    return window.__higgsLog("Vocoder already loaded: " + artifact);
  }
  window.__higgsLog("Loading vocoder: " + artifact, state.logs.length === 0);
  state.vocoderSession = await window.__higgsLoadOnnx(artifact);
  state.vocoderArtifact = artifact;
  return window.__higgsLog("Vocoder loaded: " + state.vocoderSession.inputNames.join(", "));
};

window.__higgsWavDataUrl = function(samples, sampleRate) {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const write = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, n * 2, true);
  let offset = 44;
  for (let i = 0; i < n; i++, offset += 2) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, x < 0 ? x * 32768 : x * 32767, true);
  }
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return "data:audio/wav;base64," + btoa(binary);
};

window.__higgsPlayer = function(samples, sampleRate) {
  const url = window.__higgsWavDataUrl(samples, sampleRate);
  return `<audio controls style="width:100%" src="${url}"></audio>`;
};

window.__higgsTensorIdsWithPrompt = function(encoded) {
  const state = window.__higgsState;
  const data = Array.from(encoded.input_ids.data);
  const ids = new BigInt64Array(data.length + 3);
  ids[0] = state.ttsId;
  ids[1] = state.textId;
  for (let i = 0; i < data.length; i++) ids[i + 2] = BigInt(data[i]);
  ids[ids.length - 1] = state.audioId;
  return ids;
};

window.__higgsMakeRng = function(seed) {
  let x = (Number(seed) >>> 0) || 1;
  return function() {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) / 4294967296);
  };
};

window.__higgsSampleCodebooks = function(logits, temperature, topP, topK, rng) {
  const state = window.__higgsState;
  const out = new Array(state.numCodebooks);
  const data = logits.data;
  const temp = Math.max(Number(temperature), 0);
  const kLimit = Math.floor(Number(topK));
  const pLimit = Number(topP);
  for (let cb = 0; cb < state.numCodebooks; cb++) {
    const off = cb * state.codebookSize;
    const scored = [];
    let best = 0;
    let bestVal = -Infinity;
    for (let id = 0; id < state.codebookSize; id++) {
      const v = Number(data[off + id]);
      if (v > bestVal) {
        bestVal = v;
        best = id;
      }
      scored.push([id, v]);
    }
    if (temp <= 1e-5 || kLimit === 1) {
      out[cb] = best;
      continue;
    }
    scored.sort((a, b) => b[1] - a[1]);
    let candidates = scored;
    if (kLimit > 0 && kLimit < candidates.length) {
      candidates = candidates.slice(0, kLimit);
    }
    const maxLogit = candidates[0][1];
    const probs = candidates.map(([id, v]) => [id, Math.exp((v - maxLogit) / temp)]);
    let total = probs.reduce((sum, item) => sum + item[1], 0);
    for (const item of probs) item[1] /= total;
    if (pLimit > 0 && pLimit < 1) {
      let cdf = 0;
      const kept = [];
      for (const item of probs) {
        kept.push(item);
        cdf += item[1];
        if (cdf >= pLimit) break;
      }
      total = kept.reduce((sum, item) => sum + item[1], 0);
      for (const item of kept) item[1] /= total;
      candidates.length = 0;
      for (const item of kept) candidates.push(item);
    } else {
      candidates = probs;
    }
    let r = rng();
    let chosen = candidates[candidates.length - 1][0];
    for (const [id, prob] of candidates) {
      r -= prob;
      if (r <= 0) {
        chosen = id;
        break;
      }
    }
    out[cb] = chosen;
  }
  return out;
};

window.__higgsApplyDelayMask = function(codes, step) {
  const state = window.__higgsState;
  if (step < state.numCodebooks) {
    const nextCb = step + 1;
    for (let cb = nextCb; cb < state.numCodebooks; cb++) codes[cb] = state.BOC;
  }
  return codes;
};

window.__higgsPastFeeds = function(outputs) {
  const feeds = {};
  for (let layer = 0; layer < 36; layer++) {
    feeds[`past_${layer}_key`] = outputs[`present_${layer}_key`];
    feeds[`past_${layer}_value`] = outputs[`present_${layer}_value`];
  }
  return feeds;
};

window.__higgsCodesTensor = function(codes) {
  const data = new BigInt64Array(window.__higgsState.numCodebooks);
  for (let i = 0; i < codes.length; i++) data[i] = BigInt(codes[i]);
  return new window.ort.Tensor("int64", data, [1, window.__higgsState.numCodebooks]);
};

window.__higgsDelayedToCodecRows = function(rows) {
  const state = window.__higgsState;
  const total = rows.length - (state.numCodebooks - 1);
  const frames = [];
  for (let t = 0; t < total; t++) {
    const row = [];
    let valid = true;
    for (let cb = 0; cb < state.numCodebooks; cb++) {
      const code = rows[t + cb][cb];
      if (code < 0 || code >= state.codecCodebookSize) valid = false;
      row.push(code);
    }
    if (!valid) break;
    frames.push(row);
  }
  return frames;
};

window.higgsLoadModels = async function(vocoderArtifact) {
  try {
    let [status] = await window.__higgsLoadAr();
    status = await window.__higgsLoadVocoder(vocoderArtifact);
    return [window.__higgsRenderStatus(), ""];
  } catch (error) {
    window.__higgsLog(String(error && (error.stack || error.message) || error), true);
    return [window.__higgsRenderStatus(), ""];
  }
};

window.higgsDecodeFixture = async function(vocoderArtifact) {
  try {
    const state = window.__higgsState;
    window.__higgsLog("Decoding fixture in browser...", true);
    await window.__higgsLoadVocoder(vocoderArtifact);
    const fixture = await fetch(state.repo + "sample_codes_fa_64.json").then((r) => r.json());
    const T = fixture.frames;
    const N = fixture.num_codebooks;
    const flat = new BigInt64Array(N * T);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) flat[n * T + t] = BigInt(fixture.codes_TN[t][n]);
    }
    const t0 = performance.now();
    const outputs = await state.vocoderSession.run({
      audio_codes: new window.ort.Tensor("int64", flat, [1, N, T])
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    const seconds = (outputs.audio_values.data.length / fixture.sample_rate).toFixed(2);
    const status = window.__higgsLog(`Decoded FIXED VOCODER FIXTURE: ${outputs.audio_values.data.length} samples (${seconds}s) in ${dt}s. This ignores the textbox.`);
    return [window.__higgsRenderStatus(), window.__higgsPlayer(outputs.audio_values.data, fixture.sample_rate)];
  } catch (error) {
    window.__higgsLog(String(error && (error.stack || error.message) || error), true);
    return [window.__higgsRenderStatus(), ""];
  }
};

window.higgsGenerate = async function(text, language, voice, maxSteps, temperature, topP, topK, seed, vocoderArtifact) {
  try {
    const state = window.__higgsState;
    window.__higgsLog("Starting browser TTS generation...", true);
    window.__higgsLog("Language: " + language + " / voice: " + voice);
    window.__higgsLog(`Sampler: temperature=${temperature}, top_p=${topP}, top_k=${topK}, seed=${seed}`);
    await window.__higgsLoadAr();

    const tok = await window.__higgsEnsureTokenizer();
    window.__higgsLog("Tokenizing text...");
    const encoded = await tok(String(text || "").trim(), {add_special_tokens: false});
    const ids = window.__higgsTensorIdsWithPrompt(encoded);
    const mask = new BigInt64Array(ids.length);
    mask.fill(1n);

    window.__higgsLog(`Running AR prefill: prompt length ${ids.length}...`);
    const t0 = performance.now();
    window.__higgsLog("Loading AR prefill session only...");
    state.prefillSession = await window.__higgsLoadOnnx("models/higgs_audio_v3_ar_prefill_matmul4.onnx");
    let outputs = await state.prefillSession.run({
      input_ids: new window.ort.Tensor("int64", ids, [1, ids.length]),
      attention_mask: new window.ort.Tensor("int64", mask, [1, ids.length])
    });
    await window.__higgsReleaseSession("prefillSession");
    window.__higgsLog("Released AR prefill session.");
    const rng = window.__higgsMakeRng(seed);
    let delayCount = 0;
    let eocCountdown = null;
    let done = false;
    let codes = window.__higgsSampleCodebooks(outputs.logits, temperature, topP, topK, rng);
    codes = window.__higgsApplyDelayMask(codes, delayCount);
    delayCount += 1;
    const rows = [codes.slice()];
    let past = window.__higgsPastFeeds(outputs);
    window.__higgsLog("Loading AR decode session only...");
    state.decodeSession = await window.__higgsLoadOnnx("models/higgs_audio_v3_ar_decode_matmul4.onnx");

    for (let step = 1; step < Number(maxSteps); step++) {
      if (done) break;
      outputs = await state.decodeSession.run({
        codes: window.__higgsCodesTensor(codes),
        position_ids: new window.ort.Tensor("int64", new BigInt64Array([BigInt(ids.length + step - 1)]), [1, 1]),
        ...past
      });
      codes = window.__higgsSampleCodebooks(outputs.logits, temperature, topP, topK, rng);
      if (delayCount < state.numCodebooks) {
        codes = window.__higgsApplyDelayMask(codes, delayCount);
        delayCount += 1;
      } else if (eocCountdown !== null) {
        eocCountdown -= 1;
        if (eocCountdown <= 0) done = true;
      } else if (codes[0] === state.EOC) {
        eocCountdown = state.numCodebooks <= 2 ? 0 : state.numCodebooks - 2;
      }
      rows.push(codes.slice());
      past = window.__higgsPastFeeds(outputs);
      if (step % 8 === 0) window.__higgsLog(`AR step ${step}/${maxSteps}`);
    }
    await window.__higgsReleaseSession("decodeSession");
    window.__higgsLog("Released AR decode session.");
    const arSeconds = ((performance.now() - t0) / 1000).toFixed(2);
    const codecRows = window.__higgsDelayedToCodecRows(rows);
    if (codecRows.length === 0) throw new Error("AR produced no complete codec frames.");
    window.__higgsLog(`AR produced ${codecRows.length} codec frames in ${arSeconds}s.`);

    const T = codecRows.length;
    const flat = new BigInt64Array(state.numCodebooks * T);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < state.numCodebooks; n++) flat[n * T + t] = BigInt(codecRows[t][n]);
    }
    window.__higgsLog(`Running vocoder: [1, ${state.numCodebooks}, ${T}]...`);
    await window.__higgsLoadVocoder(vocoderArtifact);
    const v0 = performance.now();
    const voc = await state.vocoderSession.run({
      audio_codes: new window.ort.Tensor("int64", flat, [1, state.numCodebooks, T])
    });
    const vocSeconds = ((performance.now() - v0) / 1000).toFixed(2);
    const audioSeconds = (voc.audio_values.data.length / state.sampleRate).toFixed(2);
    const status = window.__higgsLog(`Generated FROM TEXT. prompt_tokens=${ids.length}, delayed_rows=${rows.length}, codec_frames=${T}, audio=${audioSeconds}s, samples=${voc.audio_values.data.length}. Vocoder ${vocSeconds}s.`);
    return [window.__higgsRenderStatus(), window.__higgsPlayer(voc.audio_values.data, state.sampleRate)];
  } catch (error) {
    window.__higgsLog(String(error && (error.stack || error.message) || error), true);
    return [window.__higgsRenderStatus(), ""];
  }
};
</script>
"""


CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
#higgs-live-status {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  min-height: 360px;
  max-height: 520px;
  overflow: auto;
  padding: 12px;
  border-radius: 8px;
  background: #f6f6f6;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.45;
}
#status, #status .prose, #status .prose * {
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: break-word;
}
"""


def idle_status():
    return '<pre id="higgs-live-status">Ready. Models run in your browser with ONNX Runtime Web.</pre>'


def ping_status():
    return '<pre id="higgs-live-status">Browser JS ping requested. Waiting for client callback...</pre>', ""


def begin_generate_status():
    return '<pre id="higgs-live-status">Generate click received. Starting browser callback...</pre>', ""


with gr.Blocks(
    title="Higgs Audio v3 4-bit ONNX Browser TTS",
    css=CSS,
    head=HEAD,
    fill_width=True,
) as demo:
    gr.Markdown("# Higgs Audio v3 4-bit ONNX Browser TTS")
    gr.Markdown("Fast (Matmul4 AR transformer), with a Fast or Quality vocoder path running client-side in the browser.")

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="Text",
                value="سلام، این یک تست کوتاه برای صدای فارسی است.",
                lines=4,
                elem_id="text",
                rtl=True,
            )
            with gr.Row():
                language = gr.Dropdown(
                    label="Language",
                    choices=["Persian", "English", "Arabic", "Turkish"],
                    value="Persian",
                )
                voice = gr.Dropdown(
                    label="Voice",
                    choices=["Default zero-shot"],
                    value="Default zero-shot",
                )
            max_steps = gr.Slider(
                label="Max AR steps",
                minimum=16,
                maximum=1024,
                value=192,
                step=8,
            )
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    value=0.9,
                    step=0.05,
                )
                top_p = gr.Slider(
                    label="Top-p",
                    minimum=0.05,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                )
            with gr.Row():
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=1026,
                    value=50,
                    step=1,
                )
                seed = gr.Number(
                    label="Seed",
                    value=7,
                    precision=0,
                )
            vocoder = gr.Dropdown(
                label="Output path",
                choices=[
                    ("Fast (Matmul4 vocoder)", "models/higgs_audio_v3_vocoder_decode_matmul4.onnx"),
                    ("Quality (FP32 vocoder)", "models/higgs_audio_v3_vocoder_decode.onnx"),
                ],
                value="models/higgs_audio_v3_vocoder_decode_matmul4.onnx",
            )
            with gr.Row():
                load = gr.Button("Load models", variant="secondary")
                ping = gr.Button("Ping browser JS", variant="secondary")
                fixture = gr.Button("Test fixed vocoder fixture", variant="secondary")
                generate = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            status = gr.HTML(
                label="Status",
                value=idle_status(),
                elem_id="status",
            )
            player = gr.HTML(label="Audio")

    load.click(
        fn=None,
        inputs=[vocoder],
        outputs=[status, player],
        js="async (vocoder) => await window.higgsLoadModels(vocoder)",
        show_progress="full",
    )
    ping.click(
        fn=ping_status,
        inputs=[],
        outputs=[status, player],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=None,
        inputs=[],
        outputs=[status, player],
        js="async () => ['<pre id=\"higgs-live-status\">Browser JS is alive. Button callbacks are firing.</pre>', '']",
        show_progress="hidden",
    )
    fixture.click(
        fn=None,
        inputs=[vocoder],
        outputs=[status, player],
        js="async (vocoder) => await window.higgsDecodeFixture(vocoder)",
        show_progress="full",
    )
    generate.click(
        fn=begin_generate_status,
        inputs=[],
        outputs=[status, player],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=None,
        inputs=[text, language, voice, max_steps, temperature, top_p, top_k, seed, vocoder],
        outputs=[status, player],
        js="async (text, language, voice, maxSteps, temperature, topP, topK, seed, vocoder) => await window.higgsGenerate(text, language, voice, maxSteps, temperature, topP, topK, seed, vocoder)",
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch()
