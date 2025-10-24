#!/usr/bin/env python3
"""Extract metrics from benchmark result files and output as CSV."""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import csv


def parse_directory_name(dir_name: str) -> Dict[str, str]:
    """Parse directory name to extract model info and config."""
    # Examples:
    # - llada15_gsm8k_128
    # - llada15_gsm8k_Top-8_128
    # - llada8b_humaneval_256

    parts = dir_name.split('_')

    # Determine model type and size
    if dir_name.startswith('llada15'):
        model_type = 'LLaDA'
        model_size = '1.5'
        remaining = dir_name[7:]  # Remove 'llada15'
    elif dir_name.startswith('llada8b'):
        model_type = 'LLaDA'
        model_size = '8B'
        remaining = dir_name[7:]  # Remove 'llada8b'
    else:
        model_type = 'Unknown'
        model_size = 'Unknown'
        remaining = dir_name

    # Remove leading underscore if present
    if remaining.startswith('_'):
        remaining = remaining[1:]

    # Extract dataset, strategy, and gen_length
    parts = remaining.split('_')

    if len(parts) >= 2:
        dataset = parts[0]

        # Check if there's a strategy indicator (Top-8, Top-16, etc.)
        if len(parts) >= 3 and 'Top' in parts[1]:
            strategy = parts[1]  # Top-8, Top-16
            gen_length = parts[2]
        else:
            # No Top-k, just gen_length
            gen_length = parts[1]
            # Determine strategy from other indicators
            if len(parts) >= 3:
                # Could be cache, baseline, invert, etc.
                strategy = parts[2]
            else:
                strategy = 'Baseline'
    elif len(parts) == 1:
        dataset = parts[0]
        gen_length = 'Unknown'
        strategy = 'Baseline'
    else:
        dataset = 'Unknown'
        gen_length = 'Unknown'
        strategy = 'Baseline'

    return {
        'model_type': model_type,
        'model_size': model_size,
        'strategy': strategy,
        'dataset': dataset,
        'gen_length': gen_length
    }


def extract_metrics_from_json(json_path: Path, dataset_type: str) -> Dict[str, float]:
    """Extract relevant metrics from JSON file based on dataset type."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    metrics = {}

    # Get results section
    results = data.get('results', {})

    # Find the task key (gsm8k, humaneval, math/minerva_math, mbpp)
    task_key = None
    for key in results.keys():
        key_lower = key.lower()
        if key_lower in ['gsm8k', 'humaneval', 'math', 'mbpp']:
            task_key = key
            break
        # For MATH dataset, look for minerva_math
        if 'minerva_math' in key_lower and not any(x in key_lower for x in ['algebra', 'counting', 'geometry', 'intermediate', 'num_theory', 'prealgebra', 'precalc']):
            task_key = key
            break

    if not task_key:
        return metrics

    task_results = results[task_key]

    # Extract metrics based on dataset type
    if dataset_type == 'gsm8k':
        # GSM8K: flexible-extract exact match
        for key, value in task_results.items():
            if 'flexible-extract' in key and 'exact_match' in key and 'stderr' not in key:
                metrics['accuracy'] = value

    elif dataset_type == 'math':
        # MATH: math_verify,none (minerva math)
        for key, value in task_results.items():
            if key == 'math_verify,none':
                metrics['accuracy'] = value
                break

    elif dataset_type == 'humaneval':
        # HumanEval: pass@1,create_test
        for key, value in task_results.items():
            if 'pass@1' in key and 'create_test' in key and 'stderr' not in key:
                metrics['accuracy'] = value

    elif dataset_type == 'mbpp':
        # MBPP: pass_at_1,none
        for key, value in task_results.items():
            if key == 'pass_at_1,none':
                metrics['accuracy'] = value
                break

    # Calculate tokens/sec if we have timing information
    total_time = float(data.get('total_evaluation_time_seconds', 0))
    if total_time > 0:
        # Get number of samples and generation length
        n_samples_dict = data.get('n-samples', {})
        if task_key in n_samples_dict:
            n_samples = n_samples_dict[task_key].get('effective', 0)
        else:
            # For datasets with subtasks (e.g., MATH/minerva_math), sum up all subtasks
            n_samples = 0
            for key, value in n_samples_dict.items():
                if isinstance(value, dict) and 'effective' in value:
                    n_samples += value.get('effective', 0)

        # Try to get gen_length from model_args
        model_args = data.get('config', {}).get('model_args', '')
        gen_length_match = re.search(r'gen_length=(\d+)', model_args)
        if gen_length_match:
            gen_length = int(gen_length_match.group(1))
        else:
            gen_length = 0

        # Calculate approximate tokens/sec
        # Total tokens = n_samples * gen_length
        if n_samples > 0 and gen_length > 0:
            total_tokens = n_samples * gen_length
            tokens_per_sec = total_tokens / total_time
            metrics['tokens_per_sec'] = tokens_per_sec

    return metrics


def main():
    """Main function to extract all metrics and output CSV."""
    # Use benchmark_results directory
    base_dir = Path('benchmark_results')

    if not base_dir.exists():
        print(f"Error: {base_dir} directory not found!")
        return

    # Find all result directories
    result_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # Collect all data
    all_results = []

    for result_dir in sorted(result_dirs):
        # Parse directory name
        dir_info = parse_directory_name(result_dir.name)

        # Find JSON files
        json_files = list(result_dir.glob('**/results_*.json'))

        for json_file in json_files:
            # Extract metrics
            metrics = extract_metrics_from_json(json_file, dir_info['dataset'])

            # Combine all info
            row = {
                'Model': f"{dir_info['model_type']}-{dir_info['model_size']}",
                'Strategy': dir_info['strategy'],
                'Dataset': dir_info['dataset'].upper(),
                'Gen_Length': dir_info['gen_length'],
                'Accuracy': metrics.get('accuracy', ''),
                'Tokens_per_sec': metrics.get('tokens_per_sec', '')
            }

            all_results.append(row)

    # Write to CSV
    if all_results:
        fieldnames = ['Model', 'Strategy', 'Dataset', 'Gen_Length', 'Accuracy', 'Tokens_per_sec']

        with open('benchmark_metrics.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"Extracted metrics for {len(all_results)} experiments")
        print("Output saved to: benchmark_metrics.csv")

        # Also print to stdout in a formatted table
        print("\nResults Summary:")
        print("=" * 110)
        print(f"{'Model':<20} {'Strategy':<15} {'Dataset':<12} {'Gen_Length':<12} {'Accuracy (%)':<15} {'Tokens/sec':<15}")
        print("=" * 110)
        for row in all_results:
            acc_str = f"{row['Accuracy']*100:.2f}%" if row['Accuracy'] else "N/A"
            tok_str = f"{row['Tokens_per_sec']:.2f}" if row['Tokens_per_sec'] else "N/A"
            print(f"{row['Model']:<20} {row['Strategy']:<15} {row['Dataset']:<12} "
                  f"{row['Gen_Length']:<12} {acc_str:<15} {tok_str:<15}")
        print("=" * 110)
    else:
        print("No results found!")


if __name__ == '__main__':
    main()
