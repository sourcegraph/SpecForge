#!/usr/bin/env python3
"""
Pre-process the dataset offline to avoid distributed training timeout.
This processes the dataset once and caches it for training.
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sf_logging import get_logger

from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file

logger = get_logger(__name__)

def preprocess_dataset(args):
    """Pre-process train and eval datasets and cache them."""
    logger.info("Starting dataset pre-processing...")
    
    # Load tokenizer (same as in training script)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    processor = None
    if args.is_vlm:
        processor = AutoProcessor.from_pretrained(args.target_model)
    
    # Create cache key (same logic as training script)
    cache_params_string = (
        f"online-"
        f"{args.build_dataset_num_proc}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    
    # Check if training dataset is already cached
    processed_cache_dir = os.path.join(args.cache_dir, "processed_dataset")
    cache_files = [f for f in os.listdir(processed_cache_dir) if f.startswith(cache_key) and f.endswith('.pkl')]
    
    if cache_files:
        logger.info(f"Found {len(cache_files)} cached training dataset files - skipping processing")
        logger.info("Loading cached training dataset...")
        train_dataset = load_dataset("json", data_files=args.train_data)["train"]
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=processed_cache_dir,
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
        logger.info("Cached training dataset loaded successfully!")
    else:
        # Process training dataset from scratch
        logger.info(f"Loading training dataset from: {args.train_data}")
        train_dataset = load_dataset("json", data_files=args.train_data)["train"]
        logger.info(f"Training dataset loaded with {len(train_dataset)} examples")
        
        logger.info("Processing training dataset...")
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=processed_cache_dir,
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
    
    # Load draft config to get vocab sizes for vocab mapping
    from specforge.utils import load_config_from_file
    draft_config_path = Path(args.draft_config) if args.draft_config else None
    
    if draft_config_path and draft_config_path.exists():
        draft_model_config = load_config_from_file(str(draft_config_path))
        
        # Generate vocab mapping (only needed once from train dataset)
        logger.info("Generating vocab mapping...")
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
        logger.info(f"Vocab mapping saved to: {vocab_mapping_path}")
    else:
        logger.warning("No draft config provided, skipping vocab mapping generation")
    
    # Process evaluation dataset if provided
    if args.eval_data and Path(args.eval_data).exists():
        logger.info(f"Loading evaluation dataset from: {args.eval_data}")
        eval_dataset = load_dataset("json", data_files=args.eval_data)["train"]
        logger.info(f"Evaluation dataset loaded with {len(eval_dataset)} examples")
        
        logger.info("Processing evaluation dataset...")
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
        )
        logger.info("Evaluation dataset processing completed!")
    else:
        logger.info("No evaluation dataset provided or file not found, skipping eval preprocessing")
    
    logger.info("Dataset pre-processing completed!")
    logger.info(f"Processed datasets cached in: {os.path.join(args.cache_dir, 'processed_dataset')}")

def main():
    parser = argparse.ArgumentParser(description="Pre-process dataset for EAGLE3 training")
    
    parser.add_argument("--target-model", type=str, required=True,
                       help="Target model path")
    parser.add_argument("--train-data", type=str, required=True,
                       help="Training data JSONL file")
    parser.add_argument("--eval-data", type=str,
                       help="Evaluation data JSONL file (optional)")
    parser.add_argument("--draft-config", type=str,
                       help="Draft config JSON file (needed for vocab mapping)")
    parser.add_argument("--cache-dir", type=str, required=True,
                       help="Cache directory for processed dataset")
    parser.add_argument("--chat-template", type=str, default="qwen",
                       help="Chat template name")
    parser.add_argument("--max-length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--build-dataset-num-proc", type=int, default=8,
                       help="Number of processes for dataset building")
    parser.add_argument("--is-vlm", action="store_true",
                       help="Whether the target model is a VLM")
    parser.add_argument("--is-preformatted", action="store_true",
                       help="Whether input data is preformatted")
    
    args = parser.parse_args()
    preprocess_dataset(args)

if __name__ == "__main__":
    main()
