#!/usr/bin/env python3
"""
Run AMP Tab Benchmark for custom dataset

Usage:
python run_amp_tab_benchmark.py --num-questions 1 --data-path ~/artifacts/spec-forge/prepared_data/eval_data.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from sglang import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_jsonl(file_path: str) -> list:
    """Load JSONL file and return list of JSON objects."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main(args):
    # Select backend
    set_default_backend(select_sglang_backend(args))

    # Read custom data
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    # Prepare chat data
    chat_data = []
    for i, item in enumerate(data[:args.num_questions]):
        conversations = item.get("conversations", [])
        conversations = [cc for cc in conversations if cc['role'] in ['system', 'user']]
        
        if conversations:
            chat_data.append({"messages": conversations})
        else:
            print(f"Warning: No conversations found in sample {i}")

    print(f"Prepared {len(chat_data)} conversations for benchmarking")

    if not chat_data:
        print("No valid conversations found. Exiting.")
        return

    #####################################
    ######### SGL Program Begin #########
    #####################################
    import sglang as sgl

    @sgl.function
    def get_amp_tab_answer(s, messages):
        # Add only system and user messages (exclude assistant responses)
        for msg in messages:
            if msg["role"] == "system":
                s += sgl.system(msg["content"])
            elif msg["role"] == "user":
                s += sgl.user(msg["content"])
        
        # Generate the assistant response
        s += sgl.assistant(sgl.gen("answer", max_tokens=2048))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    print(f"Running benchmark with {len(chat_data)} conversations...")
    tic = time.perf_counter()
    states = get_amp_tab_answer.run_batch(
        chat_data,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Compute speed metrics
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    
    num_input_tokens = sum(
        s.get_meta_info("answer").get("prompt_tokens", 0) for s in states
    )
    
    total_tokens = num_output_tokens + num_input_tokens
    output_throughput = num_output_tokens / latency
    total_throughput = total_tokens / latency

    # Check for speculative decoding metrics
    has_verify = "spec_verify_ct" in states[0].get_meta_info("answer")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer")["spec_verify_ct"] for s in states
        )
        if num_verify_tokens == 0:
            accept_length = 1.0
        else:
            accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    # Collect Q&As for validation
    qa_pairs = []
    for i, (chat_item, state) in enumerate(zip(chat_data, states)):
        # Get the last user message as the "question" for display
        user_msg = ""
        for msg in reversed(chat_item["messages"]):
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        
        qa_pairs.append({
            "id": i,
            "question": user_msg,
            "response": state["answer"],
            "input_tokens": state.get_meta_info("answer").get("prompt_tokens", 0),
            "output_tokens": state.get_meta_info("answer")["completion_tokens"]
        })

    # Print results
    print("\n" + "="*50)
    print("AMP TAB BENCHMARK RESULTS")
    print("="*50)
    print(f"Dataset: {args.data_path}")
    print(f"Conversations processed: {len(chat_data)}")
    print(f"Latency: {latency:.3f} s")
    print(f"Input tokens: {num_input_tokens:,}")
    print(f"Output tokens: {num_output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Total throughput: {total_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")
    if has_verify:
        print(f"Verify tokens: {num_verify_tokens:,}")
        print("Speculative decoding: ENABLED")
    else:
        print("Speculative decoding: DISABLED")
    print("="*50)

    # Save results to JSONL
    if args.save_results:
        output_file = args.save_results
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write metadata as first line
            metadata = {
                "type": "metadata",
                "dataset": str(args.data_path),
                "num_conversations": len(chat_data),
                "latency": latency,
                "input_tokens": num_input_tokens,
                "output_tokens": num_output_tokens,
                "total_tokens": total_tokens,
                "output_throughput": output_throughput,
                "total_throughput": total_throughput,
                "accept_length": accept_length,
                "has_speculative_decoding": has_verify,
                "verify_tokens": num_verify_tokens if has_verify else 0,
                "timestamp": time.time()
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            
            # Write each Q&A pair as separate line
            for qa in qa_pairs:
                qa_record = {
                    "type": "qa_pair",
                    "id": qa["id"],
                    "messages": chat_data[qa["id"]]["messages"],  # Full conversation
                    "response": qa["response"],
                    "input_tokens": qa["input_tokens"],
                    "output_tokens": qa["output_tokens"]
                }
                f.write(json.dumps(qa_record, ensure_ascii=False) + '\n')
        
        print(f"Results saved to JSONL: {output_file}")
        print(f"  - 1 metadata record + {len(qa_pairs)} Q&A pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMP Tab benchmark on custom dataset")
    parser.add_argument("--num-questions", type=int, default=100, 
                       help="Number of conversations to process (default: 100)")
    parser.add_argument("--data-path", type=str, 
                       default="~/artifacts/spec-forge/prepared_data/eval_data.jsonl",
                       help="Path to the JSONL data file")
    parser.add_argument("--save-results", type=str, default="amp_tab_benchmark_results.jsonl",
                       help="Path to save all results (prompts + responses) as JSONL file (default: amp_tab_benchmark_results.jsonl)")
    
    args = add_common_sglang_args_and_parse(parser)
    
    # Expand home directory path
    args.data_path = Path(args.data_path).expanduser()
    
    main(args)
