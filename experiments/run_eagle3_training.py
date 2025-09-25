#!/usr/bin/env python3
"""
Run SpecForge EAGLE3 online training.
Clean wrapper around the training pipeline.
"""

import os
import argparse
import subprocess
import torch
from pathlib import Path
from datetime import datetime
import sys

# Add experiments directory to path for logging import
sys.path.insert(0, str(Path(__file__).parent))
from sf_logging import get_logger

logger = get_logger(__name__)

def get_gpu_count():
    """Auto-detect number of available GPUs."""
    try:
        return torch.cuda.device_count()
    except:
        return 1

def check_required_files(train_data, eval_data, draft_config, batch_size=1):
    """Check if required files exist and log sample counts."""
    files_to_check = [
        ("Training data", train_data),
        ("Evaluation data", eval_data), 
        ("Draft config", draft_config)
    ]
    
    for name, path in files_to_check:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Count and log samples in data files
    try:
        train_count = sum(1 for _ in open(train_data, 'r'))
        eval_count = sum(1 for _ in open(eval_data, 'r'))
        expected_steps = (train_count + batch_size - 1) // batch_size  # Ceiling division
        
        logger.info(f"Training samples found: {train_count}")
        logger.info(f"Evaluation samples found: {eval_count}")
        logger.info(f"Expected steps per epoch: {expected_steps} (with batch_size={batch_size})")
        
    except Exception as e:
        logger.warning(f"Could not count samples in data files: {e}")

def create_timestamped_output_dir(base_output_dir):
    """Create timestamped output directory with latest symlink."""
    base_path = Path(base_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = base_path / timestamp
    latest_dir = base_path / "latest"
    
    # Create timestamped directory
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {timestamped_dir}")
    
    # Update latest symlink
    if latest_dir.is_symlink() or latest_dir.exists():
        latest_dir.unlink()
    
    # Create new symlink (relative path for portability)
    latest_dir.symlink_to(timestamp)
    logger.info(f"Updated latest symlink: {latest_dir} -> {timestamp}")
    
    return str(timestamped_dir), str(latest_dir)

def setup_environment(cache_dir):
    """Setup training environment."""
    # Set cache directory (will be updated later with timestamped path)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(cache_dir) / "compiled_kernels")
    
    # Check WANDB
    if not os.environ.get('WANDB_API_KEY'):
        logger.error("WANDB_API_KEY not found in environment")
        logger.error("Set it as environment variable or in .env file")
        logger.error("Get your key at: https://wandb.ai/authorize")
        raise EnvironmentError("WANDB_API_KEY required for training")

def run_training(args):
    """Run the actual training."""
    
    logger.info("Starting SpecForge EAGLE3 training setup")
    
    # Auto-detect GPUs
    num_gpus = args.num_gpus if args.num_gpus else get_gpu_count()
    logger.info(f"Using {num_gpus} GPU(s) for training")
    
    # Check files exist
    logger.info("Validating required files...")
    check_required_files(args.train_data, args.eval_data, args.draft_config, args.batch_size)
    logger.info("All required files found")
    
    # Setup environment
    logger.info("Setting up training environment...")
    # Use a dummy cache dir for initial setup, will be overridden with timestamped one
    dummy_cache = args.cache_dir or "/tmp/dummy_cache"
    setup_environment(dummy_cache)
    
    # Create timestamped output directory  
    timestamped_output_dir, latest_output_dir = create_timestamped_output_dir(args.output_dir)
    
    # Use the provided cache directory
    shared_cache_dir = args.cache_dir
    Path(shared_cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Update cache environment for this run
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(shared_cache_dir) / "compiled_kernels")
    
    logger.info(f"Target model: {args.target_model}")
    logger.info(f"Training data: {args.train_data}")
    logger.info(f"Evaluation data: {args.eval_data}")
    logger.info(f"Output directory: {timestamped_output_dir}")
    logger.info(f"Shared cache directory: {shared_cache_dir}")
    logger.info(f"Latest symlink: {latest_output_dir}")
    
    # Find training script
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    training_script = root_dir / "scripts" / "train_eagle3_online.py"
    
    if not training_script.exists():
        logger.error(f"Training script not found: {training_script}")
        raise FileNotFoundError(f"Training script not found: {training_script}")
    
    logger.info(f"Using training script: {training_script}")
    
    # Build command
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={num_gpus}",
        str(training_script),
        "--target-model-path", args.target_model,
        "--draft-model-config", args.draft_config,
        "--train-data-path", args.train_data,
        "--eval-data-path", args.eval_data,
        "--output-dir", timestamped_output_dir,  # Use timestamped directory
        "--num-epochs", str(args.num_epochs),
        "--batch-size", str(args.batch_size),
        "--draft-global-batch-size", str(args.draft_global_batch_size),
        "--learning-rate", str(args.learning_rate),
        "--max-length", str(args.max_length),
        "--chat-template", args.chat_template,
        "--cache-dir", shared_cache_dir,
        "--embedding-key", args.embedding_key,
        "--attention-backend", args.attention_backend,
        "--ttt-length", str(args.ttt_length),
        "--tp-size", str(num_gpus),
        "--log-steps", str(args.log_steps),
        "--save-interval", str(args.save_interval),
        "--eval-interval", str(args.eval_interval),
        "--report-to", args.report_to,
        "--wandb-project", args.wandb_project,
        "--wandb-name", f"{args.wandb_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Add timestamp to wandb name
        "--dist-timeout", "60"  # 60 minutes timeout for large dataset processing
    ]
    
    # Add step-based parameters if provided
    if args.eval_steps:
        cmd.extend(["--eval-steps", str(args.eval_steps)])
    if args.save_steps:
        cmd.extend(["--save-steps", str(args.save_steps)])
    
    # Run training
    logger.info("Launching training process...")
    logger.info(f"Command: {' '.join(cmd[:3])} [training script with {len(cmd)-3} arguments]")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {timestamped_output_dir}")
        logger.info(f"Latest symlink: {latest_output_dir}")
    else:
        logger.error(f"Training failed with exit code {result.returncode}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run SpecForge EAGLE3 training")
    
    # Required arguments
    parser.add_argument("--target-model", type=str, required=True,
                       help="Target model path")
    parser.add_argument("--train-data", type=str, required=True,
                       help="Training data JSONL file")
    parser.add_argument("--eval-data", type=str, required=True,
                       help="Evaluation data JSONL file")
    parser.add_argument("--draft-config", type=str, required=True,
                       help="Draft model config JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for trained model")
    parser.add_argument("--cache-dir", type=str, required=True,
                       help="Shared cache directory for dataset and kernel caching")
    parser.add_argument("--chat-template", type=str, required=True,
                       help="Chat template name")
    
    # Training parameters
    parser.add_argument("--num-gpus", type=int,
                       help="Number of GPUs (auto-detect if not specified)")
    parser.add_argument("--num-epochs", type=int, required=True,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, required=True,
                       help="Training batch size")
    parser.add_argument("--draft-global-batch-size", type=int, required=True,
                       help="Draft model global batch size (controls gradient accumulation steps)")
    parser.add_argument("--learning-rate", type=float, required=True,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, required=True,
                       help="Maximum sequence length")
    parser.add_argument("--embedding-key", type=str, required=True,
                       help="Embedding key")
    parser.add_argument("--attention-backend", type=str, required=True,
                       help="Attention backend")
    parser.add_argument("--ttt-length", type=int, required=True,
                       help="TTT length")
    parser.add_argument("--log-steps", type=int, required=True,
                       help="Logging interval")
    parser.add_argument("--save-interval", type=int, required=True,
                       help="Save interval")
    parser.add_argument("--eval-interval", type=int, required=True,
                       help="Evaluation interval")
    parser.add_argument("--report-to", type=str, required=True,
                       help="Reporting backend")
    parser.add_argument("--wandb-project", type=str, required=True,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-name", type=str, required=True,
                       help="Weights & Biases run name")
    parser.add_argument("--eval-steps", type=int,
                       help="Run evaluation every N steps (overrides epoch-based eval)")
    parser.add_argument("--save-steps", type=int,
                       help="Save checkpoint every N steps (overrides epoch-based saving)")
    
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
