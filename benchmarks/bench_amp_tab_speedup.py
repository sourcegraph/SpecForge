#!/usr/bin/env python3
"""
Automated benchmarking script for AMP Tab dataset across different Eagle3 steps.

Usage:
python bench_amp_tab_speedup.py \
    --draft-model-paths sourcegraph/eagle3-speculator-300k-amp-tab-draft-model-acc-0p95 \
    --steps-list 1 2 4 8 16 \
    --num-questions 500 \
    --output amp_tab_speedup_results.jsonl

Note: Uses topk=1 (sequential mode), draft_tokens auto-set to steps+1 by SGLang
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AMP Tab across Eagle3 steps")
    parser.add_argument("--draft-model-paths", type=str, nargs="+", required=True,
                       help="List of draft model paths to test")
    parser.add_argument("--steps-list", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                       help="List of speculation steps to test")
    parser.add_argument("--num-questions", type=int, default=100,
                       help="Number of questions to process")
    parser.add_argument("--output", type=str, default="amp_tab_speedup_results.jsonl",
                       help="Output file for results")
    parser.add_argument("--port", type=int, default=30000,
                       help="Port for SGLang server")
    parser.add_argument("--data-path", type=str,
                       default="~/artifacts/spec-forge/prepared_data/eval_data.jsonl",
                       help="Path to eval data")
    return parser.parse_args()


def launch_sglang_server(draft_model_path: str, steps: int, port: int):
    """Launch SGLang server with specified configuration."""
    base_url = f"http://localhost:{port}"
    
    sglang_args = [
        "--model", "sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k",
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", draft_model_path,
        "--speculative-num-steps", str(steps),
        "--speculative-eagle-topk", "1",  # Always use topk=1
        "--mem-fraction-static", "0.75",
        "--tp", "1",
        "--port", str(port),
        "--trust-remote-code"
    ]
    
    process = popen_launch_server(
        "sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k",
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=sglang_args
    )
    return process


def run_benchmark(num_questions: int, data_path: str, port: int):
    """Run the AMP Tab benchmark and return results."""
    benchmark_script = Path(__file__).parent / "run_amp_tab_benchmark.py"
    
    cmd = [
        "python", str(benchmark_script),
        "--num-questions", str(num_questions),
        "--data-path", str(Path(data_path).expanduser()),
        "--port", str(port)
    ]
    
    try:
        # Capture output from benchmark script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}")
            return None
        
        # Parse output for key metrics
        lines = result.stdout.strip().split('\n')
        latency = None
        throughput = None
        accept_length = None
        
        for line in lines:
            if line.startswith("Latency:"):
                latency = float(line.split()[1])
            elif line.startswith("Output throughput:"):
                throughput = float(line.split()[2])
            elif line.startswith("Accept length:"):
                accept_length = float(line.split()[2])
        
        return {
            "latency": latency,
            "throughput": throughput,
            "accept_length": accept_length
        }
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None


def main():
    args = parse_args()
    
    results = []
    
    for draft_model_path in args.draft_model_paths:
        print(f"\n{'='*60}")
        print(f"Testing draft model: {draft_model_path}")
        print(f"{'='*60}")
        
        for steps in args.steps_list:
            print(f"\nConfig: steps={steps}, topk=1, draft_tokens={steps+1} (auto)")
            
            # Launch server
            print("Launching SGLang server...")
            process = launch_sglang_server(draft_model_path, steps, args.port)
            
            # Wait for server to be ready
            time.sleep(10)
            
            try:
                # Run benchmark
                print("Running benchmark...")
                benchmark_result = run_benchmark(args.num_questions, args.data_path, args.port)
                
                if benchmark_result:
                    result_record = {
                        "draft_model_path": draft_model_path,
                        "steps": steps,
                        "topk": 1,
                        "draft_tokens": steps + 1,  # Auto-determined by SGLang
                        "latency": benchmark_result["latency"],
                        "throughput": benchmark_result["throughput"],
                        "accept_length": benchmark_result["accept_length"],
                        "num_questions": args.num_questions,
                        "timestamp": time.time()
                    }
                    
                    results.append(result_record)
                    
                    print(f"Results: latency={benchmark_result['latency']:.3f}s, "
                          f"throughput={benchmark_result['throughput']:.1f} tok/s, "
                          f"accept_length={benchmark_result['accept_length']:.3f}")
                    
                    # Save results incrementally
                    with open(args.output, 'w') as f:
                        for record in results:
                            f.write(json.dumps(record) + '\n')
                else:
                    print("Benchmark failed!")
                    
            finally:
                # Kill server
                print("Stopping server...")
                kill_process_tree(process.pid)
                time.sleep(5)
    
    print(f"\nAll benchmarks completed! Results saved to {args.output}")
    
    # Print summary
    print(f"\nSUMMARY ({len(results)} configurations tested):")
    print("-" * 80)
    for result in results:
        model_name = Path(result["draft_model_path"]).name
        print(f"{model_name:<40} | "
              f"steps={result['steps']} (tokens={result['draft_tokens']}) | "
              f"throughput={result['throughput']:.1f} tok/s | "
              f"accept_length={result['accept_length']:.3f}")


if __name__ == "__main__":
    main()
