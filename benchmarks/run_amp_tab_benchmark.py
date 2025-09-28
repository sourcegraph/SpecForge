#!/usr/bin/env python3
"""
Run AMP Tab Benchmark for custom dataset - Essential version
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

    # Prepare chat data - filter out assistant responses
    chat_data = []
    for i, item in enumerate(data[:args.num_questions]):
        conversations = item.get("conversations", [])
        conversations = [cc for cc in conversations if cc['role'] in ['system', 'user']]
        
        if conversations:
            chat_data.append({"messages": conversations})
        else:
            print(f"Warning: No valid conversations found in sample {i}")

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
        # Add system and user messages
        for msg in messages:
            if msg["role"] == "system":
                s += sgl.system(msg["content"])
            elif msg["role"] == "user":
                s += sgl.user(msg["content"])
        
        # Generate the assistant response
        s += sgl.assistant(sgl.gen("answer", max_tokens=512))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = get_amp_tab_answer.run_batch(
        chat_data,
        temperature=0,
        max_new_tokens=512,
        num_threads=args.parallel,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    # Compute metrics
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    
    output_throughput = num_output_tokens / latency

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

    # Print results
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Accept length: {accept_length:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMP Tab benchmark on custom dataset")
    parser.add_argument("--num-questions", type=int, default=100, 
                       help="Number of conversations to process (default: 100)")
    parser.add_argument("--data-path", type=str, 
                       default="~/artifacts/spec-forge/prepared_data/eval_data.jsonl",
                       help="Path to the JSONL data file")
    
    args = add_common_sglang_args_and_parse(parser)
    
    # Expand home directory path
    args.data_path = Path(args.data_path).expanduser()
    
    main(args)
