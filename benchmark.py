import torch
import time
import argparse
from pathlib import Path

def benchmark():
    parser = argparse.ArgumentParser(description="Chatterbox Benchmark Script")
    parser.add_argument("--turbo", action="store_true", help="Benchmark the Turbo model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_turbo = args.turbo
    
    print(f"Benchmarking on {device} (Turbo: {is_turbo})...")
    
    # Load model
    start_load = time.time()
    if is_turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(device)
    else:
        from chatterbox import ChatterboxTTS
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
            if is_turbo:
                # Turbo model generate doesn't take exaggeration
                model.generate(text)
            else:
                model.generate(text, exaggeration=0.0)
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
    
    report_name = "benchmark_report_turbo.txt" if is_turbo else "benchmark_report.txt"
    with open(report_name, "w") as f:
        f.write("--- Chatterbox Inference Benchmark Report ---\n")
        f.write(f"Device: {device} (Turbo: {is_turbo})\n\n")
        f.write(f"{'Length':<12} | {'Chars':<6} | {'Time (s)':<10} | {'Chars/s':<10}\n")
        f.write("-" * 45 + "\n")
        for label, data in results.items():
            f.write(f"{label:<12} | {data['chars']:<6} | {data['time']:<10.3f} | {data['chars_per_sec']:<10.2f}\n")
    
    print(f"\nReport saved to {report_name}")

if __name__ == "__main__":
    benchmark()
