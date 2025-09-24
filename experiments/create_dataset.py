#!/usr/bin/env python3
"""
SpecForge Dataset Creator
Simple script to prepare datasets for online/offline training from base data.
"""

import json
import argparse
import subprocess
import torch
import sys
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer
from tqdm import tqdm

# Add experiments directory to path for logging import
sys.path.insert(0, str(Path(__file__).parent))
from sf_logging import get_logger

logger = get_logger(__name__)

def get_token_counts(texts: List[str], tokenizer, batch_size: int = 64) -> List[int]:
    """Get exact token counts using GPU batch tokenization."""
    if not torch.cuda.is_available():
        logger.info(f"No GPU available, using CPU tokenization...")
        return [len(tokenizer.encode(text, add_special_tokens=False)) for text in tqdm(texts, desc="CPU tokenizing")]
    
    logger.info(f"GPU tokenizing {len(texts)} conversations...")
    token_counts = []
    
    # Process in batches with proper padding for GPU
    for i in tqdm(range(0, len(texts), batch_size), desc="GPU tokenizing"):
        batch_texts = texts[i:i + batch_size]
        
        # Batch tokenize with padding (required for GPU tensors)
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False
        ).to("cuda")
        
        # Count actual tokens (excluding padding)
        for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            # Count non-padding tokens
            actual_length = attention_mask.sum().item()
            token_counts.append(actual_length)
            
    return token_counts

def extract_conversations_from_messages(example: Dict[str, Any], index: int) -> List[Dict[str, str]]:
    """Extract conversations from messages format."""
    if "messages" not in example:
        raise ValueError(f"Sample {index}: Missing 'messages' field")
        
    conversations = []
    for j, message in enumerate(example["messages"]):
        if "role" not in message:
            raise ValueError(f"Sample {index}, message {j}: Missing 'role' field")
        if "content" not in message:
            raise ValueError(f"Sample {index}, message {j}: Missing 'content' field")
            
        role = message["role"]
        content = message["content"]
        
        # Keep system, user, and assistant messages
        if role in ["system", "user", "assistant"]:
            conversations.append({"role": role, "content": content})
    
    # Must have at least user and assistant (system is optional)
    user_count = sum(1 for turn in conversations if turn["role"] == "user")
    assistant_count = sum(1 for turn in conversations if turn["role"] == "assistant")
    
    if user_count == 0:
        raise ValueError(f"Sample {index}: No user messages found")
    if assistant_count == 0:
        raise ValueError(f"Sample {index}: No assistant messages found")
    
    return conversations

def filter_by_token_length(conversations: List[Dict[str, Any]], max_length_filter: int, tokenizer) -> List[Dict[str, Any]]:
    """Filter conversations by token length using GPU batch processing."""
    if not max_length_filter:
        raise ValueError("max_length_filter must be provided for filtering")
    if not tokenizer:
        raise ValueError("tokenizer must be provided for filtering")
    if not conversations:
        raise ValueError("conversations list cannot be empty")
    
    # Extract texts for batch tokenization
    conversation_texts = []
    for conv in conversations:
        total_content = " ".join([turn["content"] for turn in conv["conversations"]])
        conversation_texts.append(total_content)
    
    token_counts = get_token_counts(conversation_texts, tokenizer)
    
    # Filter based on token counts
    converted = []
    filtered_count = 0
    
    for conv, token_count in zip(conversations, token_counts):
        if token_count <= max_length_filter:
            converted.append(conv)
        else:
            filtered_count += 1
    
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} conversations longer than {max_length_filter} tokens")
        
    return converted

def convert_to_conversations(data: List[Dict[str, Any]], max_length_filter: int = None, tokenizer=None) -> List[Dict[str, Any]]:
    """Convert messages format to SpecForge conversations format."""
    
    # First pass: Convert all messages to conversations
    all_conversations = []
    
    for i, example in enumerate(tqdm(data, desc="Processing messages")):
        try:
            conversations = extract_conversations_from_messages(example, i)
            all_conversations.append({
                "id": f"sample_{i}",
                "conversations": conversations
            })
        except ValueError as e:
            logger.error(f"Skipping sample {i}: {e}")
            continue
    
    # Second pass: Filter by token length if requested
    if max_length_filter and tokenizer:
        return filter_by_token_length(all_conversations, max_length_filter, tokenizer)
    else:
        return all_conversations

def create_dataset(args):
    """Create dataset for SpecForge training."""
    
    # Setup paths
    base_path = Path(args.base_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Base dataset: {base_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Eval samples: {args.eval_samples}")
    
    # Load tokenizer if filtering is requested
    tokenizer = None
    if args.max_length_filter and args.tokenizer_model:
        logger.info(f"Loading tokenizer from {args.tokenizer_model} for exact token counting...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            logger.warning("Skipping length filtering...")
    
    # Load and convert data
    logger.info("Loading and converting data...")
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
    logger.info(f"Converted {len(converted_data)} conversations")
    
    # Split data
    eval_size = args.eval_samples
    train_size = len(converted_data) - eval_size
    
    if eval_size >= len(converted_data):
        logger.error("eval_samples >= total converted samples")
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
    
    logger.info(f"Created train: {train_file} ({len(train_data)} samples)")
    logger.info(f"Created eval: {eval_file} ({len(eval_data)} samples)")
    
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
                logger.error(f"--{param.replace('_', '-')} is required for offline mode")
                return
                
        logger.info("Generating hidden states for offline training...")
        generate_hidden_states(
            args.model_path, 
            train_file, 
            output_dir / "hidden_states",
            args.chat_template,
            args.max_length,
            args.batch_size,
            args.mem_frac
        )
    
    logger.info(f"Dataset ready for {args.mode} training!")

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
        logger.error(f"{prepare_script} not found")
        return
    
    logger.info(f"Using {tp_size} GPU(s) for hidden states generation")
    
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
        logger.info(f"Hidden states saved to: {output_path}")
    else:
        logger.error("Failed to generate hidden states")

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
