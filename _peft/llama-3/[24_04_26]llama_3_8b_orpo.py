# %%
#!pip install -U transformers datasets accelerate peft trl bitsandbytes wandb

# %%
import gc
import os

import torch
import wandb
from datasets import load_dataset

# from google.colab import userdata
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format


# %%
os.system("nvidia-smi")

# %%
os.system("nvcc --version")

# %%
# wb_token = ""
# wandb.login(key=wb_token)

# %%
# torch.cuda.get_device_capability()[0]

# %%
if torch.cuda.get_device_capability()[0] >= 8:
    #!pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# %%
from huggingface_hub import notebook_login

# notebook_login()

# %%
# Model
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "Llama-3-8B-Orpo-v0.1"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

# %%
model

# %%
tokenizer

# %%
dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")

# note taking 4k of 40k
dataset = dataset.shuffle(seed=42).select(range(4000))


def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)

# %%
dataset

# %%
# dataset["train"][1]

# %%
orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    auto_find_batch_size=True,
    # per_device_train_batch_size=2,
    # per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(new_model)

# %%
# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

# %%
