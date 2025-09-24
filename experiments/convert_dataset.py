#!/usr/bin/env python3
"""
Convert dataset from messages format to SpecForge conversations format.
Ignores system messages and creates user-assistant pairs.
"""

import json
import uuid
import argparse
from typing import Dict, Any

def convert_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert single example from messages to conversations format."""
    if "messages" not in example:
        raise ValueError("Missing 'messages' key")
        
    conversations = []
    for message in example["messages"]:
        role = message.get("role")
        content = message.get("content", "")
        
        if role in ["user", "assistant"]:
            conversations.append({"role": role, "content": content})
            
    return {
        "id": str(uuid.uuid4()),
        "conversations": conversations
    }

def convert_dataset(input_path: str, output_path: str) -> None:
    """Convert entire dataset."""
    converted_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            converted = convert_example(example)
            converted_data.append(converted)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in converted_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
    print(f"Converted {len(converted_data)} examples")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to SpecForge format")
    parser.add_argument("input_path", help="Input JSONL file path")
    parser.add_argument("output_path", help="Output JSONL file path")
    
    args = parser.parse_args()
    convert_dataset(args.input_path, args.output_path)
