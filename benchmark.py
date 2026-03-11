import time
import torch
from chatterbox import ChatterboxTTS
import numpy as np

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}...")
    
    # Load model
    start_load = time.time()
    model = ChatterboxTTS.from_pretrained(device)
    end_load = time.time()
    print(f"Model loaded in {end_load - start_load:.2f}s")

    test_cases = {
        "20 chars": "Hello, how are you today? This is a test.",
        "60 chars": "The quick brown fox jumps over the lazy dog. Programming is a very fun and creative activity.",
        "300 chars": (
            "Chatterbox is an open-source text-to-speech model by Resemble AI. "
            "It is designed to be high-quality, efficient, and easy to use for developers. "
            "In this benchmark, we are testing the inference speed of the model across different text lengths to see how well it scales. "
            "Performance is crucial for real-time applications like virtual assistants and gaming. "
            "Let's see how it performs today!"
        )
    }

    results = {}

    # Warmup
    print("Warming up...")
    model.generate("Warmup text")
    
    for label, text in test_cases.items():
        print(f"Processing {label} ({len(text)} chars)...")
        start = time.time()
        with torch.no_grad():
            model.generate(text)
        end = time.time()
        elapsed = end - start
        results[label] = {
            "time": elapsed,
            "chars": len(text),
            "chars_per_sec": len(text) / elapsed
        }
        print(f"Done in {elapsed:.2f}s")

    print("\n--- Benchmark Report ---")
    print(f"{'Length':<12} | {'Chars':<6} | {'Time (s)':<10} | {'Chars/s':<10}")
    print("-" * 45)
    for label, data in results.items():
        print(f"{label:<12} | {data['chars']:<6} | {data['time']:<10.3f} | {data['chars_per_sec']:<10.2f}")
    
    with open("benchmark_report.txt", "w") as f:
        f.write("--- Chatterbox Inference Benchmark Report ---\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"{'Length':<12} | {'Chars':<6} | {'Time (s)':<10} | {'Chars/s':<10}\n")
        f.write("-" * 45 + "\n")
        for label, data in results.items():
            f.write(f"{label:<12} | {data['chars']:<6} | {data['time']:<10.3f} | {data['chars_per_sec']:<10.2f}\n")

if __name__ == "__main__":
    benchmark()
