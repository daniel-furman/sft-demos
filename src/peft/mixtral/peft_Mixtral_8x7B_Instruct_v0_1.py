# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

#  [markdown]
# ## Finetuned on 4x A6000s
#
# We will leverage PEFT library from Hugging Face ecosystem, as well as QLoRA for more memory efficient finetuning

# %%
os.system("nvcc --version")

# %%
# %%
os.system("nvidia-smi")

# %%
#  [markdown]
# ## Setup
#
# Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

#
# os.system("yes | pip uninstall protobuf")
# os.system("pip install -q -U accelerate protobuf datasets bitsandbytes einops wandb sentencepiece protobuf==3.20.* transformers")
# os.system("pip install -q -U git+https://github.com/huggingface/peft")
# os.system("pip install -q -U git+https://github.com/huggingface/trl")

# %%
# %%
import torch
import pandas as pd
import tqdm
import numpy as np
import copy
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)

# %%
# [markdown]
# Let's also load the tokenizer below

# %%
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1", use_fast=True, trust_remote_code=True
)
tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST] ' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
tokenizer.pad_token = tokenizer.unk_token
tokenizer.clean_up_tokenization_spaces = True
tokenizer.add_bos_token = False
tokenizer.padding_side = "right"
tokenizer.pad_token

# %%
# [markdown]
# ## Dataset
# %%
seed = 42

# %%
# grab the first 40000 entries of SlimOrca in an instruction format

dataset_name = "Open-Orca/SlimOrca"
print(f"\nLoading {dataset_name} dataset...")
dataset_SlimOrca = load_dataset(dataset_name, split="train", streaming=True)

dataset_SlimOrca = dataset_SlimOrca.take(40000)
texts = []

for row in dataset_SlimOrca:
    messages_keep = []
    for message in row["conversations"]:
        if message["from"] == "human":
            messages_keep.append({"role": "user", "content": message["value"]})
        if message["from"] == "gpt":
            messages_keep.append({"role": "assistant", "content": message["value"]})

    text = tokenizer.apply_chat_template(
        messages_keep, tokenize=False, add_generation_prompt=False
    )
    texts.append(text)

pandas_dataset_SlimOrca = pd.DataFrame([texts]).T
pandas_dataset_SlimOrca.columns = ["text"]
pandas_dataset_SlimOrca

# %%
# grab first 40000 rows of platypus in an instruction format

dataset_name = "garage-bAInd/Open-Platypus"
print(f"\nLoading {dataset_name} dataset...")
dataset_platypus = load_dataset(dataset_name, split="train", streaming=True)

dataset_platypus = dataset_platypus.take(40000)
texts = []

for row in dataset_platypus:
    messages_keep = []
    messages_keep.append({"role": "user", "content": row["instruction"]})
    messages_keep.append({"role": "assistant", "content": row["output"]})
    text = tokenizer.apply_chat_template(
        messages_keep, tokenize=False, add_generation_prompt=False
    )
    texts.append(text)

pandas_dataset_platypus = pd.DataFrame([texts]).T
pandas_dataset_platypus.columns = ["text"]
pandas_dataset_platypus

# %%
# grab first 40000 rows of platypus in an instruction format

dataset_name = "jondurbin/airoboros-2.2.1"
print(f"\nLoading {dataset_name} dataset...")
dataset_airoboros = load_dataset(dataset_name, split="train", streaming=True)

dataset_airoboros = dataset_airoboros.take(40000)
texts = []

for row in dataset_airoboros:
    messages_keep = []
    messages_keep.append({"role": "user", "content": row["instruction"]})
    messages_keep.append({"role": "assistant", "content": row["response"]})
    text = tokenizer.apply_chat_template(
        messages_keep, tokenize=False, add_generation_prompt=False
    )
    texts.append(text)

pandas_dataset_airoboros = pd.DataFrame([texts]).T
pandas_dataset_airoboros.columns = ["text"]
pandas_dataset_airoboros

# %%
pandas_train_dataset = pd.concat(
    [pandas_dataset_platypus, pandas_dataset_SlimOrca, pandas_dataset_airoboros]
).reset_index(drop=True)
pandas_train_dataset

# %%
train_dataset = Dataset.from_pandas(pandas_train_dataset)
train_dataset

# remove old text cols
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in ["text"]]
)

print("Final train dataset:")
print(train_dataset)
print(train_dataset[0])
print(train_dataset[-1])

# %%
print(tokenizer.decode(tokenizer.encode(train_dataset[-1]["text"])))

# %%
encoded_train_dataset = train_dataset.map(
    lambda examples: tokenizer(examples["text"]), batched=True
)
encoded_train_dataset

# %%
encoded_train_dataset

# %%
num_dropped = 0
rows_to_drop = []
max_num_tokens_taken = []
for i in tqdm.tqdm(range(len(pandas_train_dataset))):
    row = encoded_train_dataset[i]
    num_tokens = len(row["input_ids"])
    if num_tokens > 500:
        rows_to_drop.append(i)
        num_dropped += 1
    else:
        max_num_tokens_taken.append(num_tokens)

num_dropped

# %%
np.max(max_num_tokens_taken)

# %%
# pandas_train_dataset

# %%
pandas_train_dataset = pandas_train_dataset.drop(rows_to_drop).reset_index(drop=True)

# %%
pandas_train_dataset

# %%
train_dataset = Dataset.from_pandas(pandas_train_dataset)

print("Train dataset:")
train_dataset = train_dataset.shuffle(seed=seed)
print(train_dataset)
print(train_dataset[0])
print(train_dataset[-1])

# %%
len(tokenizer.encode(train_dataset[100]["text"]))

# %%
# ensure none over 500 tokens

# check that above worked
lens = []
encoded_train_dataset = train_dataset.map(
    lambda examples: tokenizer(examples["text"]), batched=True
)
for row in encoded_train_dataset:
    lens.append(len(row["input_ids"]))
np.max(lens)

# %%
# remove old text cols
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in ["text"]]
)

# %%
# [markdown]
# ## Loading the model
# [markdown]

# %%
model_name = "mistralai/Mixtral-8x7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False


# %%
model

# %%
# [markdown]
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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# [markdown]
# ## Loading the trainer
# [markdown]
# Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# %%
output_dir = "./results"
num_train_epochs = 2
auto_find_batch_size = True
gradient_accumulation_steps = 2
optim = "paged_adamw_32bit"
save_strategy = "epoch"
learning_rate = 3e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
logging_strategy = "steps"
logging_steps = 25
evaluation_strategy = "no"
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
    evaluation_strategy=evaluation_strategy,
    bf16=bf16,
)

# %%
# [markdown]
# Then finally pass everthing to the trainer

# %%
max_seq_length = 512

response_template = "[/INST]"
print(f"Response template for collator: {response_template}")
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer, mlm=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    data_collator=collator,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# [markdown]
# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training

# %%
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# %%
# [markdown]
# ## Train the model
# [markdown]
# Now let's train the model! Simply call `trainer.train()`

# %%
trainer.train()

# wandb: Syncing run charmed-sun-100
# wandb: ⭐️ View project at https://wandb.ai/dryanfurman/huggingface
# wandb: 🚀 View run at https://wandb.ai/dryanfurman/huggingface/runs/2e0wfvff

# %%
print(model)
