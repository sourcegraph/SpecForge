#!/usr/bin/env python3
"""
Analyze target model for SpecForge training.
Downloads and inspects the model configuration.
"""

import json
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download

# Target model configuration
TARGET_MODEL = "sourcegraph/amp-tab-v3-all-comb-no-pred-neg-0p20p-rel-qwen-chat-pred-3k"
# Use base model for analysis since target model is private
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Base model to understand architecture

def download_model(model_name: str):
    """Download model to local cache."""
    print(f"Downloading model: {model_name}")
    local_path = snapshot_download(repo_id=model_name)
    print(f"Model cached at: {local_path}")
    return local_path

def analyze_model_config(model_name: str):
    """Analyze model configuration."""
    print(f"\nAnalyzing model: {model_name}")
    
    # Load configuration
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Architecture: {config.architectures}")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    
    if hasattr(config, 'num_key_value_heads'):
        print(f"Number of KV heads: {config.num_key_value_heads}")
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Chat template available: {tokenizer.chat_template is not None}")
    
    return config, tokenizer

def check_specforge_compatibility(config):
    """Check SpecForge compatibility requirements."""
    print(f"\nSpecForge Compatibility Check:")
    
    # Check if model type is supported
    supported_models = ["qwen2", "llama", "phi3"]  # Based on SpecForge target models
    model_type = config.model_type.lower()
    
    is_supported = any(supported in model_type for supported in supported_models)
    print(f"Model type '{model_type}' supported: {is_supported}")
    
    if "qwen" in model_type:
        print("✅ Qwen model detected - should work with SpecForge Qwen target models")
        return "qwen2"  # or qwen3 based on version
    elif "llama" in model_type:
        print("✅ Llama model detected - should work with SpecForge Llama target models")
        return "llama"
    else:
        print("⚠️ Model type may need custom SpecForge target model implementation")
        return model_type

def generate_training_config(config, model_type):
    """Generate draft model configuration for training."""
    print(f"\nSuggested Draft Model Configuration:")
    
    # Draft model should be much smaller
    draft_config = {
        "architectures": ["LlamaForCausalLMEagle3"],  # SpecForge uses Llama architecture for draft
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size if hasattr(config, 'intermediate_size') else config.hidden_size * 4,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, 'num_key_value_heads', config.num_attention_heads),
        "num_hidden_layers": 1,  # Draft model typically has 1 layer
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "model_type": "llama",
        "torch_dtype": "float16",
        "use_cache": True
    }
    
    print(f"Draft model layers: {draft_config['num_hidden_layers']}")
    print(f"Target model layers: {config.num_hidden_layers}")
    print(f"Hidden size: {draft_config['hidden_size']}")
    print(f"Vocab size: {draft_config['vocab_size']}")
    
    return draft_config

if __name__ == "__main__":
    print(f"Target model (private): {TARGET_MODEL}")
    print(f"Analyzing base model: {BASE_MODEL}")
    
    # Download and analyze base model to understand architecture
    local_path = download_model(BASE_MODEL)
    config, tokenizer = analyze_model_config(BASE_MODEL)
    
    # Check compatibility
    model_type = check_specforge_compatibility(config)
    
    # Generate draft config
    draft_config = generate_training_config(config, model_type)
    
    # Save draft config
    output_path = "/home/ronaksagtani/dev/SpecForge/experiments/draft_model_config.json"
    with open(output_path, 'w') as f:
        json.dump(draft_config, f, indent=2)
    
    print(f"\nDraft model config saved: {output_path}")
    print(f"Configuration based on {BASE_MODEL} architecture")
    print(f"Should work with target model: {TARGET_MODEL}")
    
    # Print SpecForge training command template
    print(f"\nSpecForge training setup:")
    print(f"  Target model: {TARGET_MODEL}")
    print(f"  Chat template: qwen")
    print(f"  Draft config: {output_path}")
