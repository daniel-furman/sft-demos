# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

# %% [markdown]
# ## Finetune Falcon-180b on 4x A6000s
#
# Welcome to this LambdaLabs notebook that shows how to fine-tune the recent Falcon-180b model on 4x A6000s.
#
# We will leverage PEFT library from Hugging Face ecosystem, as well as QLoRA for more memory efficient finetuning

# %%
os.system("nvcc --version")

# %%
os.system("nvidia-smi")

# %% [markdown]
# ## Setup
#
# Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

# %%
# os.system("pip install -q -U trl transformers accelerate protobuf==3.19.0")
# os.system("pip install -q datasets bitsandbytes einops wandb")
# os.system("pip install -q git+https://github.com/huggingface/peft")


# %%
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)


# %%
os.system("pip list")

# %% [markdown]
# ## Dataset
#

# %%
seed = 42


# %%
# grab the first 20000 entries of dolphin in an instruction format

dataset_name = "ehartford/dolphin"
print(f"\nLoading {dataset_name} dataset...")
dataset_dolphin = load_dataset(dataset_name, split="train", streaming=True)

dataset_dolphin = dataset_dolphin.take(7500)
questions = []
responses = []

for row in dataset_dolphin:
    questions.append(f'{row["instruction"]} {row["input"]}')
    responses.append(row["output"])

pandas_dataset_dolphin = pd.DataFrame([questions, responses]).T
pandas_dataset_dolphin.columns = ["prompt", "response"]

# grab all rows of platypus in an instruction format

dataset_name = "garage-bAInd/Open-Platypus"
print(f"\nLoading {dataset_name} dataset...")
dataset_platypus = load_dataset(dataset_name, split="train", streaming=True)
dataset_platypus = dataset_platypus.take(5000)

questions = []
responses = []

for row in dataset_platypus:
    questions.append(
        f'You are a helpful AI assistant. Write a response that appropriately completes the request. {row["instruction"]}'
    )
    responses.append(row["output"])

pandas_dataset_platypus = pd.DataFrame([questions, responses]).T
pandas_dataset_platypus.columns = ["prompt", "response"]

pandas_train_dataset = pd.concat([pandas_dataset_platypus, pandas_dataset_dolphin])


train_dataset = Dataset.from_pandas(pandas_train_dataset.iloc[0:10000, :])
train_dataset

# remove old text cols
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in ["prompt", "response"]]
)

print("Print an example in the train dataset:")
print(train_dataset)
print(train_dataset[0])

print("Final train dataset:")
train_dataset = train_dataset.shuffle(seed=seed)
print(train_dataset)
print(train_dataset[0])
print(train_dataset[-1])

eval_dataset = Dataset.from_pandas(pandas_train_dataset.iloc[10000:, :])
# remove old text cols
eval_dataset = eval_dataset.remove_columns(
    [col for col in eval_dataset.column_names if col not in ["prompt", "response"]]
)

print("Print an example in the eval dataset:")
print(eval_dataset)
print(eval_dataset[0])

print("Final eval dataset:")
eval_dataset = eval_dataset.shuffle(seed=seed)
print(eval_dataset)
print(eval_dataset[0])
print(eval_dataset[-1])

# let's now write a function to format the dataset for instruction fine-tuning


def formatting_prompts_func(dataset):
    instructions = []
    for i in range(len(dataset["prompt"])):
        text = f"{dataset['prompt'][i]}\n{dataset['response'][i]}"
        instructions.append(text)
    return instructions


# %% [markdown]
# ## Loading the model
# %% [markdown]
# In this section we will load the Falcon-180b quantize it in 4bit and attach LoRA adapters on it. Let's get started!

# %%
model_name = "tiiuae/falcon-180B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_auth_token=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False


# %%
model

# %% [markdown]
# Let's also load the tokenizer below

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# Below we will load the configuration file in order to create the LoRA model. According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. Therefore we will add `q_proj`, `k_proj`, `v_proj`, `o_proj` layers in the target modules.

# %%
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
)

# %% [markdown]
# ## Loading the trainer
# %% [markdown]
# Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# %%
output_dir = "./results"
num_train_epochs = 1
auto_find_batch_size = True
gradient_accumulation_steps = 1
optim = "paged_adamw_32bit"
save_strategy = "epoch"
learning_rate = 2e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
logging_strategy = "steps"
logging_steps = 25
do_eval = False
evaluation_strategy = "steps"
prediction_loss_only = True
eval_steps = 0.2
bf16 = True

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    auto_find_batch_size=auto_find_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,
    do_eval=do_eval,
    # evaluation_strategy=evaluation_strategy,
    prediction_loss_only=prediction_loss_only,
    eval_steps=eval_steps,
    bf16=bf16,
)

# %% [markdown]
# Then finally pass everthing to the trainer

# %%
max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# %% [markdown]
# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training

# %%
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# %% [markdown]
# ## Train the model
# %% [markdown]
# Now let's train the model! Simply call `trainer.train()`

# %%
trainer.train()

# wandb: Currently logged in as: dryanfurman. Use `wandb login --relogin` to force relogin
# wandb: Tracking run with wandb version 0.15.5
# wandb: Run `wandb offline` to turn off syncing.
# wandb: Syncing run vital-microwave-35 to Weights & Biases (docs)
# wandb: ⭐️ View project at https://wandb.ai/dryanfurman/huggingface
# wandb: 🚀 View run at View run at https://wandb.ai/dryanfurman/huggingface/runs/7ey9zxp9
