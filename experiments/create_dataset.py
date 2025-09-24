#!/usr/bin/env python3
"""
SpecForge Dataset Creator
Simple script to prepare datasets for online/offline training from base data.
"""

import json
import argparse
import subprocess
import torch
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer

def get_exact_token_count(text: str, tokenizer) -> int:
    """Get exact token count using the model's tokenizer."""
    return len(tokenizer.encode(text))

def convert_to_conversations(data: List[Dict[str, Any]], max_length_filter: int = None, tokenizer=None) -> List[Dict[str, Any]]:
    """Convert messages format to SpecForge conversations format."""
    converted = []
    
    for i, example in enumerate(data):
        if "messages" not in example:
            raise ValueError(f"Sample {i}: Missing 'messages' field")
            
        conversations = []
        for j, message in enumerate(example["messages"]):
            if "role" not in message:
                raise ValueError(f"Sample {i}, message {j}: Missing 'role' field")
            if "content" not in message:
                raise ValueError(f"Sample {i}, message {j}: Missing 'content' field")
                
            role = message["role"]
            content = message["content"]
            
            if role in ["user", "assistant"]:
                conversations.append({"role": role, "content": content})
        
        if len(conversations) >= 2:
            # Filter by length if specified
            if max_length_filter and tokenizer:
                total_content = " ".join([turn["content"] for turn in conversations])
                exact_tokens = get_exact_token_count(total_content, tokenizer)
                
                if exact_tokens > max_length_filter:
                    continue  # Skip this conversation
            
            converted.append({
                "id": f"sample_{i}",
                "conversations": conversations
            })
    
    return converted

def create_dataset(args):
    """Create dataset for SpecForge training."""
    
    # Setup paths
    base_path = Path(args.base_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Base dataset: {base_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Samples: {args.num_samples}")
    print(f"🔍 Eval samples: {args.eval_samples}")
    
    # Load tokenizer if filtering is requested
    tokenizer = None
    if args.max_length_filter and args.tokenizer_model:
        print(f"🔧 Loading tokenizer from {args.tokenizer_model} for exact token counting...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
            print(f"✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer: {e}")
            print("Skipping length filtering...")
    
    # Load and convert data
    print("\n⚙️  Loading and converting data...")
    data = []
    with open(base_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # If num_samples is None or "None", process all data
            if args.num_samples != "None" and args.num_samples is not None and i >= args.num_samples:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    converted_data = convert_to_conversations(data, args.max_length_filter, tokenizer)
    print(f"✅ Converted {len(converted_data)} conversations")
    
    # Split data
    eval_size = args.eval_samples
    train_size = len(converted_data) - eval_size
    
    if eval_size >= len(converted_data):
        print("❌ Error: eval_samples >= total converted samples")
        return
    
    train_data = converted_data[:train_size]
    eval_data = converted_data[train_size:train_size + eval_size]
    
    # Save datasets
    train_file = output_dir / "train_data.jsonl"
    eval_file = output_dir / "eval_data.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        for example in eval_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created train: {train_file} ({len(train_data)} samples)")
    print(f"✅ Created eval: {eval_file} ({len(eval_data)} samples)")
    
    # Generate hidden states for offline mode
    if args.mode == "offline":
        # Check required offline parameters
        offline_params = [
            ("model_path", "Model path"),
            ("chat_template", "Chat template"), 
            ("max_length", "Max length"),
            ("batch_size", "Batch size"),
            ("mem_frac", "Memory fraction")
        ]
        
        for param, name in offline_params:
            if not getattr(args, param):
                print(f"❌ Error: --{param.replace('_', '-')} is required for offline mode")
                return
                
        print(f"\n⚙️  Generating hidden states for offline training...")
        generate_hidden_states(
            args.model_path, 
            train_file, 
            output_dir / "hidden_states",
            args.chat_template,
            args.max_length,
            args.batch_size,
            args.mem_frac
        )
    
    print(f"\n🎉 Dataset ready for {args.mode} training!")

def generate_hidden_states(
    model_path: str, 
    data_path: Path, 
    output_path: Path,
    chat_template: str,
    max_length: int,
    batch_size: int,
    mem_frac: float
):
    """Generate hidden states for offline training."""
    
    # Auto-detect GPUs
    try:
        num_gpus = torch.cuda.device_count()
    except:
        num_gpus = 1
    
    tp_size = min(num_gpus, 4) if num_gpus > 1 else 1
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find the SpecForge script
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    prepare_script = root_dir / "scripts" / "prepare_hidden_states.py"
    
    if not prepare_script.exists():
        print(f"❌ Error: {prepare_script} not found")
        return
    
    print(f"🔧 Using {tp_size} GPU(s) for hidden states generation")
    
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={tp_size}",
        str(prepare_script),
        "--model-path", model_path,
        "--enable-aux-hidden-states",
        "--data-path", str(data_path),
        "--chat-template", chat_template,
        "--max-length", str(max_length),
        "--tp-size", str(tp_size),
        "--batch-size", str(batch_size),
        f"--mem-frac={mem_frac}",
        "--output-path", str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ Hidden states saved to: {output_path}")
    else:
        print(f"❌ Failed to generate hidden states")

def main():
    parser = argparse.ArgumentParser(
        description="Create SpecForge datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Online training dataset (500 samples, 100 for eval)
  python create_dataset.py --mode online --num-samples 500 --eval-samples 100
  
  # Offline training dataset with hidden states
  python create_dataset.py --mode offline --num-samples 1000 --eval-samples 200 \\
    --model-path MODEL_PATH --chat-template TEMPLATE --max-length LENGTH \\
    --batch-size BATCH --mem-frac FRAC
  
  # Custom paths
  python create_dataset.py --base-dataset /path/to/data.jsonl \\
    --output-dir /path/to/output --mode online
        """
    )
    
    parser.add_argument("--mode", choices=["online", "offline"], required=True,
                       help="Training mode: online or offline")
    parser.add_argument("--base-dataset", type=str, required=True,
                       help="Path to base dataset JSONL file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for prepared dataset")
    parser.add_argument("--num-samples", required=True,
                       help="Total number of samples to process (use 'None' for all samples)")
    parser.add_argument("--eval-samples", type=int, required=True,
                       help="Number of samples for evaluation")
    parser.add_argument("--model-path", type=str,
                       help="Model path for hidden states generation (required for offline mode)")
    parser.add_argument("--chat-template", type=str,
                       help="Chat template for hidden states generation (required for offline mode)")
    parser.add_argument("--max-length", type=int,
                       help="Maximum sequence length for processing (required for offline mode)")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size for hidden states generation (required for offline mode)")
    parser.add_argument("--mem-frac", type=float,
                       help="Memory fraction for hidden states generation (required for offline mode)")
    parser.add_argument("--max-length-filter", type=int,
                       help="Filter out conversations longer than this token count (optional)")
    parser.add_argument("--tokenizer-model", type=str,
                       help="Model path for tokenizer (used for exact token counting in filtering)")
    
    args = parser.parse_args()
    
    # Convert num_samples to int if it's not None
    if args.num_samples != "None":
        try:
            args.num_samples = int(args.num_samples)
            # Validate arguments
            if args.eval_samples >= args.num_samples:
                print("❌ Error: eval_samples must be less than num_samples")
                return
        except ValueError:
            print("❌ Error: num_samples must be an integer or 'None'")
            return
    
    if not Path(args.base_dataset).exists():
        print(f"❌ Error: Base dataset not found: {args.base_dataset}")
        return
    
    create_dataset(args)

if __name__ == "__main__":
    main()
