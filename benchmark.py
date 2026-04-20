import time
import os
import sys
import json
import psutil
import statistics
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from yecoai_cognitive_layer import FeatureEngine, CognitiveModel
except ImportError:
    print("Error: Could not import yecoai_cognitive_layer. Make sure you are in the project root.")
    sys.exit(1)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_speed_benchmark(engine, model, num_iterations=100):
    print(f"\n--- Speed Benchmark ---")
    test_texts = [
        "A user is requesting system performance evaluation under normal conversational load.",
        "Loop detected in system processing pipeline. Loop detected in system processing pipeline.",
        "An extremely expanded input block designed to stress token processing " * 50,
        "98765 " * 100
    ]
    latencies = []
    total_tokens = 0
    for _ in range(num_iterations):
        for text in test_texts:
            total_tokens += len(text.split())
            start = time.perf_counter()
            v, f = engine.extract_features(text)
            model.predict(v, f)
            latencies.append((time.perf_counter() - start) * 1000)
    
    avg_lat = statistics.mean(latencies)
    avg_tokens = total_tokens / (num_iterations * len(test_texts))
    lat_1k = (avg_lat / avg_tokens) * 1000
    
    print(f"  Avg Latency:          {avg_lat:.4f} ms")
    print(f"  Latency per 1k tokens: {lat_1k:.2f} ms")
    return avg_lat, lat_1k

def run_detection_benchmark(engine, model):
    print(f"\n--- Detection & Accuracy Benchmark (25 Cases) ---")
    test_cases = [
        {"prompt": "Compose a short mountain-themed verse", "text": "Snowy ridges touch the clouds while silent winds pass through valleys.", "expected": "Normal", "cat": "Normal (Poetry)"},
        {"prompt": "Tell a medical humor joke", "text": "The patient said second opinion, doctor replied 'you also need glasses'.", "expected": "Normal", "cat": "Normal (Joke)"},
        {"prompt": "Describe plant energy conversion", "text": "Plants convert light energy into chemical energy using chlorophyll.", "expected": "Normal", "cat": "Normal (Science)"},
        {"prompt": "Write a basic python snippet", "text": "def run():\n    return 'ok'", "expected": "Normal", "cat": "Normal (Code)"},
        {"prompt": "Casual greeting response", "text": "I'm good, thanks. Let me know what you need.", "expected": "Normal", "cat": "Normal (Chat)"},
        {"prompt": "Summarize Orwell novel", "text": "A dystopian world where surveillance controls every aspect of life.", "expected": "Normal", "cat": "Normal (Literature)"},
        {"prompt": "Capital city query", "text": "Paris is the administrative capital of France.", "expected": "Normal", "cat": "Normal (Fact)"},

        {"prompt": "Repeat keyword stress test", "text": "alpha " * 20, "expected": "Loop", "cat": "Loop (Token)"},
        {"prompt": "Narrative repetition test", "text": "The system starts again. " * 5, "expected": "Loop", "cat": "Loop (Phrase)"},
        {"prompt": "Numeric repetition sequence", "text": "9 8 7 9 8 7 9 8 7 9 8 7", "expected": "Loop", "cat": "Loop (Pattern)"},
        {"prompt": "System loop simulation", "text": "running... running... running... running...", "expected": "Loop", "cat": "Loop (System)"},
        {"prompt": "Repeated intent phrase", "text": "I should leave now. I should leave now. I should leave now.", "expected": "Loop", "cat": "Loop (Intent)"},
        {"prompt": "Symbol spam", "text": "@@@@ @@@@ @@@@ @@@@", "expected": "Loop", "cat": "Loop (Symbols)"},
        {"prompt": "Binary repetition", "text": "10101010101010101010101010101010", "expected": "Loop", "cat": "Loop (Binary)"},

        {"prompt": "Car engine question", "text": "Learning a new language requires consistent daily practice and immersion.", "expected": "Amnesia", "cat": "Amnesia (Shift)"},
        {"prompt": "Define ML", "text": "Blue sky rotates slowly under warm digital silence of fragmented ideas.", "expected": "Amnesia", "cat": "Amnesia (Noise)"},
        {"prompt": "History topic", "text": "Boil water first before adding rice to ensure proper cooking texture.", "expected": "Amnesia", "cat": "Amnesia (Mismatch)"},
        {"prompt": "Translation request", "text": "Tomorrow the weather might change depending on atmospheric pressure.", "expected": "Amnesia", "cat": "Amnesia (Drift)"},
        {"prompt": "Formal writing task", "text": "A small bird flew across the endless corridor of forgotten memories.", "expected": "Amnesia", "cat": "Amnesia (Style break)"},
        {"prompt": "Biography question", "text": "Desk lamp phone keyboard paper bottle window chair.", "expected": "Amnesia", "cat": "Amnesia (List)"},

        {"prompt": "Explain loop concept", "text": "A loop repeats execution while a condition evaluates to true.", "expected": "Normal", "cat": "FP (Concept)"},
        {"prompt": "Encouragement cheer", "text": "Keep going! Keep going! Keep going!", "expected": "Normal", "cat": "FP (Cheer)"},
        {"prompt": "Echo instruction", "text": "Learning is fun. Learning is fun. Learning is fun.", "expected": "Normal", "cat": "FP (Echo)"},
        {"prompt": "Poetic repetition", "text": "Wind moves trees, wind moves trees, softly across hills.", "expected": "Normal", "cat": "FP (Poetry)"},
        {"prompt": "Short numeric echo", "text": "42 42 42", "expected": "Normal", "cat": "FP (Echo)"}
    ]
    
    print(f"{'Category':<25} | {'Expected':<10} | {'Actual':<10} | {'Result':<10}")
    print("-" * 65)
    
    correct = 0
    for case in test_cases:
        v, f = engine.extract_features(case['text'], prompt=case['prompt'])
        pred, _ = model.predict(v, f)
        res = "PASS" if pred == case['expected'] else "FAIL"
        if res == "PASS": correct += 1
        print(f"{case['cat']:<25} | {case['expected']:<10} | {pred:<10} | {res:<10}")
        
    accuracy = (correct / len(test_cases)) * 100
    print(f"\n  Total Accuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})")
    return accuracy

def main():
    print(f"========================================")
    print(f"   YecoAI Cognitive Layer Benchmark v1.0.0")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"========================================")
    
    start_mem = get_memory_usage()
    engine = FeatureEngine()
    weights_path = os.path.join("src", "yecoai_cognitive_layer", "weights.json")
    model = CognitiveModel.load_from_json(weights_path)
    
    run_speed_benchmark(engine, model)
    acc = run_detection_benchmark(engine, model)
    
    total_ram = get_memory_usage()
    print(f"\n--- Final Resource Usage ---")
    print(f"  Total RAM Usage: {total_ram:.2f} MB")
    print(f"========================================\n")

if __name__ == "__main__":
    main()