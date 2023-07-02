"""
## Finetune an instruction-following LLM

This Python script shows how to finetune an instruction-following MPT model on a single H100 GPU (80 GB). 

We use "mosaicml/mpt-7b" as the base model and an instruction dataset derived from "timdettmers/openassistant-guanaco" for the train set (all open-source and licensed for commercial use).

We will leverage the Hugging Face ecosystem for supervised finetuning (sft) with the handy [sft_trainer](https://huggingface.co/docs/trl/main/en/sft_trainer) function. 

At the end of the script, we will have a finetuned instruction-following model cached to disk that we can then upload to a private model repo on the Hugging Face hub (see post-process-sft-llm.ipynb). 

### Reproducibility

Cluster info: This script was executed on an Ubuntu instance with an H100 GPU (80 GB) running on [Lambda Labs](https://lambdalabs.com/) (cluster type = gpu_1x_h100_pcie). 

Runtime: Each epoch takes roughly 45 min. Lambda Labs's rate for the gpu_1x_h100_pcie cluster is 1.99 dollars/hour. Thus, the finetuning is quite cost-effective. 

### Warning

This script will only run on a workstation with 1 NVIDIA GPU and is not compatible with Multi-GPU training.
"""

import os
import time

start = time.time()

os.system("nvidia-smi")

"""
## Setup

Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `transformers`, `datasets` and `trl` to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will also install `einops` as it is a requirement to load MPT models, as well as `triton_pre_mlir` for triton optimized attention.
"""

os.system("pip install -q -U trl transformers accelerate datasets einops")
os.system(
    "pip install -q -U triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
)
os.system("pip list")

# import libraries

import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset
from trl import SFTTrainer

# print GPU available memory

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
print("max_memory: ", max_memory)

"""
## Dataset

For our experiment, we will use the `timdettmers/openassistant-guanaco` dataset to train general purpose instruct model.

The dataset can be found [here](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
"""

dataset_name = "Open-Orca/OpenOrca"
print(f"\nLoading {dataset_name} dataset...")
dataset_orca = load_dataset(dataset_name, split="train", streaming=True)
prompts = []
responses = []
dataset_orca

# grab the firt 10k entries in an instruction format
dataset_head = dataset_orca.take(10000)
ids = []
system_prompts = []
questions = []
responses = []

for row in dataset_head:
    ids.append(row["id"])
    system_prompts.append(row["system_prompt"])
    questions.append(row["question"])
    responses.append(row["response"])

pandas_dataset_orca = pd.DataFrame([ids, system_prompts, questions, responses]).T
pandas_dataset_orca.columns = ["id", "system_prompt", "prompt", "response"]
dataset_orca = Dataset.from_pandas(pandas_dataset_orca)

# remove old text cols
dataset_orca = dataset_orca.remove_columns(
    [col for col in dataset_orca.column_names if col not in ["prompt", "response"]]
)

print("Print an example in the train dataset:")
print(dataset_orca)
print(dataset_orca[0])

print("Final train dataset:")
train_dataset = dataset_orca.shuffle(seed=42)
print(train_dataset)
print(train_dataset[0])
print(train_dataset[-1])

# let's now write a function to format the dataset for instruction fine-tuning
# we will use the mpt-instruct model docs format
# see https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts for docs


def formatting_prompts_func(dataset):
    instructions = []
    for i in range(len(dataset["prompt"])):
        text = f"{dataset['prompt'][i]}\n{dataset['response'][i]}"
        instructions.append(text)
    return instructions


"""
## Loading the model

In this section we will load the [MPT-7B model](https://huggingface.co/mosaicml/mpt-7b).
"""

# load assets

model_id = "mosaicml/mpt-7b"

# mpt tokenizer load
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# set mpt tokenizer padding token to eos token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"{model_id} tokenizer model_max_length: ", tokenizer.model_max_length)

# mpt llm load
config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# custom options
config.attn_config["attn_impl"] = "triton"  # Optional triton attention
config.init_device = "cuda"  # For fast initialization directly on GPU!
config.max_seq_len = tokenizer.model_max_length
config.torch_dtype = "bfloat16"  # Set bfloat16 data type for sft

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
print("max memory: ", max_memory)

"""

## Loading the trainer

Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets. Let's first load the training arguments below.
from transformers import TrainingArguments
# see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
"""

output_dir = "./results"
num_train_epochs = 3
auto_find_batch_size = True
gradient_accumulation_steps = 1
optim = "adamw_torch"
save_strategy = "epoch"
learning_rate = 5e-5
lr_scheduler_type = "constant"
logging_strategy = "steps"
logging_steps = 50


training_arguments = transformers.TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    auto_find_batch_size=auto_find_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    logging_strategy=logging_strategy,
    logging_steps=logging_steps,
)

"""
Then finally pass everything to the trainer
"""

max_seq_length = tokenizer.model_max_length

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

"""
## Train the model

Now let's train the model! Simply call `trainer.train()`
"""

trainer.train()

# finished: print GPU available memory and total time
free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
print("max memory: ", max_memory)

end = time.time()
print("total time: ", end - start)
