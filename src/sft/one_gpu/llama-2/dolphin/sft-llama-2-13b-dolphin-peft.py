# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Finetune Llama-2-13b on an A6000
# 
# Welcome to this LambdaLabs notebook that shows how to fine-tune the recent Llama-2-13b model on a single GPU.
# 
# We will leverage PEFT library from Hugging Face ecosystem, as well as QLoRA for more memory efficient finetuning

# %%
get_ipython().system('nvcc --version')


# %%
get_ipython().system('nvidia-smi')

# %% [markdown]
# ## Setup
# 
# Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

# %%
get_ipython().system('pip install -q -U trl transformers accelerate protobuf==3.19.0')
get_ipython().system('pip install -q datasets bitsandbytes einops wandb')
get_ipython().system('pip install -q git+https://github.com/huggingface/peft')


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
get_ipython().system('pip list')

# %% [markdown]
# ## Dataset
# 
# For our experiment, we will use the `ehartford/dolphin` dataset to train general purpose instruct model.
# 
# 
# The dataset can be found [here](https://huggingface.co/datasets/ehartford/dolphin)
# 

# %%
seed = 42


# %%
dataset_name = "ehartford/dolphin"
print(f"\nLoading {dataset_name} dataset...")
dataset_dolphin = load_dataset(dataset_name, split="train", streaming=True)

# grab the first 110000 entries in an instruction format
dataset_head = dataset_dolphin.take(110000)
questions = []
responses = []

for row in dataset_head:
    questions.append(f'{row["instruction"]} {row["input"]}')
    responses.append(row["output"])

pandas_dataset_dolphin = pd.DataFrame([questions, responses]).T
pandas_dataset_dolphin.columns = ["prompt", "response"]

dataset_dolphin_train = Dataset.from_pandas(
    pandas_dataset_dolphin.iloc[0:100000, :]
)
# remove old text cols
dataset_dolphin_train = dataset_dolphin_train.remove_columns(
    [
        col
        for col in dataset_dolphin_train.column_names
        if col not in ["prompt", "response"]
    ]
)

print("Print an example in the train dataset:")
print(dataset_dolphin_train)
print(dataset_dolphin_train[0])

print("Final train dataset:")
train_dataset = dataset_dolphin_train.shuffle(seed=seed)
print(train_dataset)
print(train_dataset[0])
print(train_dataset[-1])

dataset_dolphin_eval = Dataset.from_pandas(pandas_dataset_dolphin.iloc[100000:, :])
# remove old text cols
dataset_dolphin_eval = dataset_dolphin_eval.remove_columns(
    [
        col
        for col in dataset_dolphin_eval.column_names
        if col not in ["prompt", "response"]
    ]
)

print("Print an example in the eval dataset:")
print(dataset_dolphin_eval)
print(dataset_dolphin_eval[0])

print("Final eval dataset:")
eval_dataset = dataset_dolphin_eval.shuffle(seed=seed)
print(eval_dataset)
print(eval_dataset[0])
print(eval_dataset[-1])

# let's now write a function to format the dataset for instruction fine-tuning
# we will use the mpt-instruct model docs format
# see https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts for docs

def formatting_prompts_func(dataset):
    instructions = []
    for i in range(len(dataset["prompt"])):
        text = f"{dataset['prompt'][i]}\n{dataset['response'][i]}"
        instructions.append(text)
    return instructions

# %% [markdown]
# ## Loading the model
# %% [markdown]
# In this section we will load the [Falcon 7B model](https://huggingface.co/tiiuae/falcon-7b), quantize it in 4bit and attach LoRA adapters on it. Let's get started!

# %%
from huggingface_hub import login

login()


# %%
get_ipython().system('git config --global credential.helper store')


# %%
model_name = "meta-llama/Llama-2-13b-hf"

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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
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
lr_scheduler_type = "linear"
warmup_ratio = 0.03
logging_strategy = "steps"
logging_steps = 25
do_eval = True
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
    # do_eval=do_eval,
    # evaluation_strategy=evaluation_strategy,
    prediction_loss_only=prediction_loss_only,
    eval_steps=eval_steps,
    bf16=bf16,
)

# %% [markdown]
# Then finally pass everthing to the trainer

# %%
max_seq_length = 4096

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
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

# Tracking run with wandb version 0.15.5
# Run data is saved locally in /home/ubuntu/wandb/run-20230723_071649-pohixish
# Syncing run dulcet-blaze-13 to Weights & Biases (docs)
# View project at https://wandb.ai/dryanfurman/huggingface
# View run at https://wandb.ai/dryanfurman/huggingface/runs/pohixish

# %% [markdown]
# During training, the model should converge nicely as follows:
# 
# ![image](https://raw.githubusercontent.com/daniel-furman/sft-demos/main/assets/jul_22_23_3_15_00_log_loss_curves_llama-2-7b-guanaco.png)
# 
# The `SFTTrainer` also takes care of properly saving only the adapters during training instead of saving the entire model.
# %% [markdown]
# ## Upload model to HF repo

# %%
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# peft_model_id = "results/checkpoint-4923"
# peft_model_id = "dfurman/llama-2-13b-guanaco-peft"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# %%
# push to hub
model_id_load = "dfurman/llama-2-13b-guanaco-peft"

# tokenizer
tokenizer.push_to_hub(model_id_load, use_auth_token=True)
# safetensors
model.push_to_hub(model_id_load, use_auth_token=True, safe_serialization=True)
# torch tensors
model.push_to_hub(model_id_load, use_auth_token=True)


# %%
# text generation function


def llama_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: int = 1.0,
) -> str:
    """
    Initialize the pipeline
    Uses Hugging Face GenerationConfig defaults
        https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig
    Args:
        model (transformers.AutoModelForCausalLM): Falcon model for text generation
        tokenizer (transformers.AutoTokenizer): Tokenizer for model
        prompt (str): Prompt for text generation
        max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
        temperature (float, optional): The value used to modulate the next token probabilities.
            Defaults to 1.0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(
        device
    )  # tokenize inputs, load on device

    # when running Torch modules in lower precision, it is best practice to use the torch.autocast context manager.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        response = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded_output = tokenizer.decode(
        response["sequences"][0],
        skip_special_tokens=True,
    )  # grab output in natural language

    return decoded_output[len(prompt) :]  # remove prompt from output


# %%
import tqdm
import time

prompt = "You are a helpful assistant. Write me a long list of things to do in San Francisco:\n"

runtimes = []
for i in tqdm.tqdm(range(25)):
    start = time.time()
    response = llama_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=50,
        temperature=0.92,
    )
    end = time.time()
    runtimes.append(end - start)
    assert len(tokenizer.encode(response)) == 52


# %%
avg_runtime = torch.mean(torch.tensor(runtimes)).item()
print(f"Runtime avg in seconds: {avg_runtime}")  # time in seconds


# %%



