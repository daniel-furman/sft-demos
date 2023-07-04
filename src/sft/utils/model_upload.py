import argparse
import os

os.system(
    "pip install -q -U safetensors huggingface_hub transformers accelerate sentencepiece einops"
)
os.system("huggingface-cli login")
os.system("git config --global credential.helper store")

import torch
import transformers


def upload_llm(args):
    if "flan" in args.input_model_path:
        tokenizer = transformers.T5Tokenizer.from_pretrained(args.input_model_path)
        model = transformers.T5ForConditionalGeneration.from_pretrained(
            args.input_model_path, device_map="auto"
        )

    model.push_to_hub(
        args.output_model_path, use_auth_token=True, safe_serialization=True
    )
    tokenizer.push_to_hub(args.output_model_path, use_auth_token=True)


def main():
    parser = argparse.ArgumentParser(
        description="Simple example of a single GPU training script."
    )
    parser.add_argument(
        "--input_model_path",
        type=str,
        help="Input model path, can be HF repo or local folder.",
        required=True,
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        help="Output model path to HF repo.",
        required=True,
    )
    args = parser.parse_args()
    upload_llm(args)


if __name__ == "__main__":
    main()
