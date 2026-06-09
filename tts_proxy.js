const express = require('express');
const axios = require('axios');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// ==========================================
// CONFIGURATION
// ==========================================
const COMFY_URL = 'http://127.0.0.1:8188';
const CHATTERBOX_URL = 'http://127.0.0.1:8090'; // Original TTS server
const OUTPUT_DIR = 'E:/CUIPP/ComfyUI/output';
const WORKFLOW_PATH = 'E:/CUIPP/tts-workflow.json';
const PORT = 8091; // Clients should hit this port for OpenAI compatibility

// Verify workflow exists
if (!fs.existsSync(WORKFLOW_PATH)) {
    console.error(`[Error] Workflow file not found at ${WORKFLOW_PATH}`);
    process.exit(1);
}
const workflowTemplate = JSON.parse(fs.readFileSync(WORKFLOW_PATH, 'utf8'));
 
// ==========================================
// PROXY DISCOVERY ENDPOINTS
// ==========================================
 
// Forward models list
app.get('/v1/models', async (req, res) => {
    try {
        const response = await axios.get(`${CHATTERBOX_URL}/v1/models`);
        res.json(response.data);
    } catch (e) {
        res.status(500).json({ error: "Failed to proxy /v1/models", message: e.message });
    }
});
 
// Forward voices list
app.get(['/v1/voices', '/v1/audio/voices'], async (req, res) => {
    try {
        const response = await axios.get(`${CHATTERBOX_URL}/v1/audio/voices`);
        res.json(response.data);
    } catch (e) {
        res.status(500).json({ error: "Failed to proxy /v1/voices", message: e.message });
    }
});
 
// Forward capabilities
app.get(['/chatterbox/capabilities', '/v1/audio/capabilities'], async (req, res) => {
    try {
        const response = await axios.get(`${CHATTERBOX_URL}/chatterbox/capabilities`);
        res.json(response.data);
    } catch (e) {
        res.status(500).json({ error: "Failed to proxy capabilities", message: e.message });
    }
});
 
// Forward presets
app.get('/chatterbox/presets', async (req, res) => {
    try {
        const response = await axios.get(`${CHATTERBOX_URL}/chatterbox/presets`);
        res.json(response.data);
    } catch (e) {
        res.status(500).json({ error: "Failed to proxy presets", message: e.message });
    }
});

app.post('/v1/audio/speech', async (req, res) => {
    const startTime = Date.now();
    try {
        const { input, model, voice } = req.body;
        
        if (!input) {
            return res.status(400).json({ error: "Missing 'input' field in request body." });
        }

        console.log(`[Proxy] Request: "${input.substring(0, 60)}${input.length > 60 ? '...' : ''}"`);

        // 1. CLONE WORKFLOW & INJECT DATA
        const workflow = JSON.parse(JSON.stringify(workflowTemplate));
        const filename = `proxy_tts_${Date.now()}_${Math.floor(Math.random() * 1000)}.ogg`;

        // Update Node 1 (Post Request Node) - inject text into str0 and voice into str1
        if (workflow["1"]) {
            workflow["1"].inputs.str0 = input;
            workflow["1"].inputs.str1 = voice || 'ivy';
        } else {
            throw new Error("Node 1 not found in workflow template.");
        }
        
        // Update Node 7 (SaveRawBytes) - set unique filename
        if (workflow["7"]) {
            workflow["7"].inputs.filename = filename;
        } else {
            throw new Error("Node 7 not found in workflow template.");
        }

        // 2. SEND TO COMFYUI
        console.log(`[Proxy] Queuing job in ComfyUI...`);
        const promptResponse = await axios.post(`${COMFY_URL}/prompt`, { prompt: workflow });
        const promptId = promptResponse.data.prompt_id;
        
        // 3. POLL FOR COMPLETION
        console.log(`[Proxy] Prompt ID: ${promptId}. Polling for output...`);
        let completed = false;
        const timeout = 90000; // 90 seconds (TTS can be slow)
        const pollInterval = 1000;
        const startPolling = Date.now();

        while (Date.now() - startPolling < timeout) {
            const historyResponse = await axios.get(`${COMFY_URL}/history/${promptId}`);
            if (historyResponse.data[promptId]) {
                completed = true;
                break;
            }
            await new Promise(r => setTimeout(r, pollInterval));
        }

        if (!completed) {
            throw new Error("ComfyUI generation timed out after 90 seconds.");
        }

        // 4. READ RESULTING FILE
        const filePath = path.join(OUTPUT_DIR, filename);
        
        // Tiny sleep to ensure OS has finished writing the file
        await new Promise(r => setTimeout(r, 300));

        if (!fs.existsSync(filePath)) {
            throw new Error(`Output file not found: ${filePath}`);
        }

        const audioBuffer = fs.readFileSync(filePath);
        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        
        console.log(`[Proxy] Finished! Sending ${audioBuffer.length} bytes (Time: ${duration}s)`);

        // 5. RESPOND WITH AUDIO
        res.set({
            'Content-Type': 'audio/ogg',
            'Content-Length': audioBuffer.length,
            'X-Generation-Time': `${duration}s`,
            'Access-Control-Allow-Origin': '*'
        });
        
        res.send(audioBuffer);

        // 6. CLEANUP (Optional - keeping disk clean)
        try {
            fs.unlinkSync(filePath);
        } catch (e) {
            console.warn(`[Proxy] Cleanup warning: ${e.message}`);
        }

    } catch (error) {
        const msg = error.response?.data?.error || error.message;
        console.error(`[Proxy] 🔥 Error: ${msg}`);
        res.status(500).json({ error: msg });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🤖 ComfyUI TTS Proxy Server`);
    console.log(`----------------------------------------`);
    console.log(`🟢 Status: Running`);
    console.log(`🔗 API: http://localhost:${PORT}/v1/audio/speech`);
    console.log(`🔗 Metadata: http://localhost:${PORT}/v1/models`);
    console.log(`🔗 Voices: http://localhost:${PORT}/v1/audio/voices`);
    console.log(`⚙️  Target (TTS): ${COMFY_URL}`);
    console.log(`⚙️  Target (Meta): ${CHATTERBOX_URL}`);
    console.log(`📁 Watch: ${OUTPUT_DIR}`);
    console.log(`----------------------------------------\n`);
});
