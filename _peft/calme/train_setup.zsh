pip3 install -U transformers datasets accelerate peft trl bitsandbytes wandb
huggingface-cli login
huggingface-cli download MaziyarPanahi/calme-2.4-rys-78b
python3 [24_09_23]train_calme_78b_orpo.py