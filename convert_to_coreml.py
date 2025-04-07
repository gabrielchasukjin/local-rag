#!/usr/bin/env python3
"""
Utility script to convert Llama-3.1-8B-Instruct to CoreML format
This script requires macOS with Apple Silicon and the coremltools package
"""

import os
import argparse
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Llama model to CoreML format")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name or path of the model to convert"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the CoreML model"
    )
    parser.add_argument(
        "--compute_units",
        type=str,
        default="ALL",
        choices=["CPU_ONLY", "CPU_AND_GPU", "ALL"],
        help="Compute units for the CoreML model"
    )
    return parser.parse_args()

def convert_model(model_name, output_dir, compute_units):
    print(f"Converting {model_name} to CoreML format...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model and tokenizer
    print("Loading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Define the input and output shapes
    input_shape = {'input_ids': (1, 256)}  # Batch size 1, sequence length 256
    
    # Define the compute precision
    compute_precision = ct.precision.FLOAT16
    
    # Compute unit mapping
    compute_unit_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "ALL": ct.ComputeUnit.ALL
    }
    
    # Prepare a sample input
    sample_text = "Hello, I'm a language model"
    sample_input = tokenizer(sample_text, return_tensors="pt")
    
    # Convert to CoreML format
    print("Converting model to CoreML format. This may take a while...")
    
    # First, trace the model with PyTorch
    traced_model = torch.jit.trace(
        model,
        example_inputs=(sample_input.input_ids,),
        check_trace=False
    )
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=input_shape["input_ids"],
                dtype=np.int32
            )
        ],
        outputs=[
            ct.TensorType(
                name="output_ids",
                dtype=np.int32
            )
        ],
        compute_units=compute_unit_map[compute_units],
        compute_precision=compute_precision,
        convert_to="mlprogram"
    )
    
    # Save the model
    output_path = os.path.join(output_dir, f"{model_name.split('/')[-1]}.mlpackage")
    mlmodel.save(output_path)
    
    print(f"Model converted successfully and saved to {output_path}")
    return output_path

def main():
    args = parse_args()
    
    if not torch.backends.mps.is_available():
        print("Warning: Apple MPS (Metal Performance Shaders) is not available.")
        print("This script works best on Apple Silicon Macs.")
    
    try:
        output_path = convert_model(args.model_name, args.output_dir, args.compute_units)
        print("\nTo use this model with the RAG system, make sure to update the model_path in CoreMLLlama class.")
        print(f"Set model_path to: {output_path}")
    except Exception as e:
        print(f"Error converting model: {e}")
        print("\nAlternatively, you can download pre-converted CoreML models from:")
        print("https://developer.apple.com/metal/llama/")

if __name__ == "__main__":
    main() 