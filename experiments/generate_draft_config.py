#!/usr/bin/env python3
"""
Generate proper draft model config for the target model.
"""

import json
import argparse
from transformers import AutoConfig
from pathlib import Path

def generate_draft_config(target_model_path: str, output_path: str):
    """Generate draft config matching target model parameters."""
    
    print(f"Loading target model config: {target_model_path}")
    target_config = AutoConfig.from_pretrained(target_model_path)
    
    print(f"Target model type: {target_config.model_type}")
    print(f"Target hidden size: {target_config.hidden_size}")
    print(f"Target layers: {target_config.num_hidden_layers}")
    print(f"Target vocab size: {target_config.vocab_size}")
    print(f"Target max position: {target_config.max_position_embeddings}")
    
    # Create draft config based on target model
    draft_config = {
        "architectures": ["LlamaForCausalLMEagle3"],  # EAGLE3 always uses Llama architecture
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": target_config.bos_token_id,
        "eos_token_id": target_config.eos_token_id,
        "head_dim": target_config.hidden_size // target_config.num_attention_heads,
        "hidden_act": target_config.hidden_act,
        "hidden_size": target_config.hidden_size,  # Match target
        "initializer_range": target_config.initializer_range,
        "intermediate_size": target_config.intermediate_size,  # Match target
        "max_position_embeddings": min(target_config.max_position_embeddings, 32768),  # Use target's but cap for efficiency
        "max_window_layers": target_config.num_hidden_layers,  # Match target layers
        "model_type": "llama",  # EAGLE3 architecture requirement
        "num_attention_heads": target_config.num_attention_heads,  # Match target
        "num_hidden_layers": 1,  # Draft model speed optimization - only 1 layer!
        "num_key_value_heads": target_config.num_key_value_heads,  # Match target
        "rms_norm_eps": target_config.rms_norm_eps,
        "rope_scaling": target_config.rope_scaling,
        "rope_theta": target_config.rope_theta,
        "sliding_window": target_config.sliding_window,
        "tie_word_embeddings": target_config.tie_word_embeddings,
        "torch_dtype": str(target_config.torch_dtype).replace("torch.", ""),
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": getattr(target_config, 'use_sliding_window', False),
        "vocab_size": target_config.vocab_size,  # Match target exactly
        "draft_vocab_size": target_config.vocab_size  # Use same vocab (safer without mapping infrastructure)
    }
    
    # Add pad_token_id if it exists
    if hasattr(target_config, 'pad_token_id') and target_config.pad_token_id is not None:
        draft_config["pad_token_id"] = target_config.pad_token_id
    
    # Save config
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(draft_config, f, indent=2)
    
    print(f"\n✅ Generated draft config: {output_file}")
    print(f"Draft layers: {draft_config['num_hidden_layers']} (vs target: {target_config.num_hidden_layers})")
    print(f"Draft vocab: {draft_config['vocab_size']} (matches target)")
    print(f"Draft max_pos: {draft_config['max_position_embeddings']} (from target: {target_config.max_position_embeddings})")
    
    return draft_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate draft model config from target model")
    parser.add_argument("--target-model", type=str, required=True,
                       help="Target model path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output config file path")
    
    args = parser.parse_args()
    generate_draft_config(args.target_model, args.output)
