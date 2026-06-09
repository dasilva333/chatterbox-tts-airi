---
title: Higgs Audio v3 ONNX Browser
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
license: other
---

# Higgs Audio v3 ONNX Browser

Browser-only ONNX Runtime Web demo for the Higgs Audio v3 vocoder component.

The app loads `Reza2kn/Higgs-Audio-v3-TTS-4bit-ONNX` directly in the browser and
decodes a real Persian Higgs codebook fixture to audio client-side.

This is not the full text-to-speech graph yet: the autoregressive Higgs/Qwen3
text-to-codebook generator still needs a separate ONNX export and JavaScript
sampling loop.
