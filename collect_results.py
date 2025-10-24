import json
import os
from pathlib import Path

results_dir = Path("benchmark_results")
output_file = "benchmark_summary.csv"

results = []

models = {
    "llada8b": "LLaDA-8B-Instruct",
    "llada15": "LLaDA-1.5",
    "fastdllm8b": "Fast-dLLM-8B",
    "fastdllm15": "Fast-dLLM-1.5"
}

benchmarks = ["gsm8k", "math", "humaneval", "mbpp"]
gen_lengths = [128, 256]

for model_key, model_name in models.items():
    for benchmark in benchmarks:
        for gen_length in gen_lengths:
            result_path = results_dir / f"{model_key}_{benchmark}_{gen_length}" / "results.json"
            
            if result_path.exists():
                with open(result_path, 'r') as f:
                    data = json.load(f)
                    
                    if 'results' in data:
                        task_results = data['results']
                        
                        for task_name, metrics in task_results.items():
                            if isinstance(metrics, dict):
                                acc_key = None
                                for key in ['exact_match', 'acc', 'pass@1', 'accuracy']:
                                    if key in metrics:
                                        acc_key = key
                                        break
                                
                                if acc_key:
                                    accuracy = metrics[acc_key] * 100 if metrics[acc_key] < 1 else metrics[acc_key]
                                    
                                    results.append({
                                        'Model': model_name,
                                        'Benchmark': benchmark.upper(),
                                        'Gen_Length': gen_length,
                                        'Accuracy': f"{accuracy:.1f}%",
                                    })

with open(output_file, 'w') as f:
    f.write("Model,Benchmark,Gen_Length,Accuracy\n")
    for r in results:
        f.write(f"{r['Model']},{r['Benchmark']},{r['Gen_Length']},{r['Accuracy']}\n")

print(f"Results collected in {output_file}")
print("\nSummary:")
for r in results:
    print(f"{r['Model']:20} | {r['Benchmark']:10} | Gen={r['Gen_Length']:3} | {r['Accuracy']}")
