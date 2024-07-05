pip install -U transformers datasets accelerate peft trl bitsandbytes wandb
huggingface-cli login
huggingface-cli download Qwen/Qwen2-72B-Instruct
python [24_07_03]qwen_2_72b_orpo.py