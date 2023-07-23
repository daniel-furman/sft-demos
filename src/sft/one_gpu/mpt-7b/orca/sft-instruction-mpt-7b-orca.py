"""
## Finetune an instruction-following LLM

This Python script shows how to finetune an instruction-following MPT model on a single H100 GPU (80 GB). 

We use "mosaicml/mpt-7b" as the base model and an instruction dataset derived from "ehartford/dolphin" for the train set (all open-source and licensed for commercial use).

We will leverage the Hugging Face ecosystem for supervised finetuning (sft) with the handy [sft_trainer](https://huggingface.co/docs/trl/main/en/sft_trainer) function. 

At the end of the script, we will have a finetuned instruction-following model cached to disk that we can then upload to a model repo on the Hugging Face hub. 

### Reproducibility

Cluster info: This script was executed on an Ubuntu instance with an H100 GPU (80 GB) running on [Lambda Labs](https://lambdalabs.com/) (cluster type = gpu_1x_h100_pcie). 

Runtime: The script takes roughly 45 min to complete 10,000 ehartford/dolphin examples. Lambda Labs's rate for the gpu_1x_h100_pcie cluster is 1.99 dollars/hour. Thus, the finetuning is quite cost-effective. 

### Warning

This script will only run on a workstation with 1 NVIDIA GPU and is not compatible with Multi-GPU training.
"""

import os
import time
import argparse

start = time.time()

os.system("nvidia-smi")

"""
## Setup

Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `transformers`, `datasets` and `trl` to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will also install `einops` as it is a requirement to load MPT models, as well as `triton_pre_mlir` for triton optimized attention.
"""

os.system("pip install -q -U trl transformers accelerate datasets einops pandas")
os.system(
    "pip install -q -U triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python"
)
os.system("pip list")

# import libraries

import torch
import transformers
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import pandas as pd


def training_function(args):
    lr = args.lr
    num_epochs = args.num_epochs
    seed = args.seed

    transformers.set_seed(seed)

    # print GPU available memory

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    print("Max VRAM (GB): ", max_memory)

    """
    ## Dataset
    
    For our experiment, we will use the `ehartford/dolphin` dataset to train general purpose instruct model.
    
    The dataset can be found [here](https://huggingface.co/datasets/ehartford/dolphin)
    """

    dataset_name = "ehartford/dolphin"
    print(f"\nLoading {dataset_name} dataset...")
    dataset_orca = load_dataset(dataset_name, split="train", streaming=True)

    # grab the first 110000 entries in an instruction format
    dataset_head = dataset_orca.take(110000)
    questions = []
    responses = []

    for row in dataset_head:
        questions.append(f'{row["instruction"]} {row["input"]}')
        responses.append(row["output"])

    pandas_dataset_orca = pd.DataFrame([questions, responses]).T
    pandas_dataset_orca.columns = ["prompt", "response"]

    dataset_orca_train = Dataset.from_pandas(pandas_dataset_orca.iloc[0:100000, :])
    # remove old text cols
    dataset_orca_train = dataset_orca_train.remove_columns(
        [
            col
            for col in dataset_orca_train.column_names
            if col not in ["prompt", "response"]
        ]
    )

    print("Print an example in the train dataset:")
    print(dataset_orca_train)
    print(dataset_orca_train[0])

    print("Final train dataset:")
    train_dataset = dataset_orca_train.shuffle(seed=seed)
    print(train_dataset)
    print(train_dataset[0])
    print(train_dataset[-1])

    dataset_orca_eval = Dataset.from_pandas(pandas_dataset_orca.iloc[100000:, :])
    # remove old text cols
    dataset_orca_eval = dataset_orca_eval.remove_columns(
        [
            col
            for col in dataset_orca_eval.column_names
            if col not in ["prompt", "response"]
        ]
    )

    print("Print an example in the eval dataset:")
    print(dataset_orca_eval)
    print(dataset_orca_eval[0])

    print("Final eval dataset:")
    eval_dataset = dataset_orca_eval.shuffle(seed=seed)
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
    print(f"{model_id} tokenizer eos_token: ", tokenizer.eos_token)
    print(f"{model_id} tokenizer pad_token: ", tokenizer.pad_token)
    print(f"{model_id} tokenizer model_max_length: ", tokenizer.model_max_length)

    # mpt llm load
    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # custom options
    config.attn_config["attn_impl"] = "triton"  # Optional triton attention
    config.init_device = "meta"  # For fast initialization
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
    print("Max VRAM (GB): ", max_memory)

    """
    
    ## Loading the trainer
    
    Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets. Let's first load the training arguments below.
    from transformers import TrainingArguments
    # see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    """

    output_dir = "./results"
    num_train_epochs = num_epochs
    auto_find_batch_size = True
    gradient_accumulation_steps = 1
    optim = "adamw_torch"
    save_strategy = "epoch"
    learning_rate = lr
    lr_scheduler_type = "linear"
    warmup_ratio = 0.03
    logging_strategy = "steps"
    logging_steps = 50
    do_eval = True
    evaluation_strategy = "steps"
    prediction_loss_only = True
    eval_steps = 0.2

    training_arguments = transformers.TrainingArguments(
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
        evaluation_strategy=evaluation_strategy,
        prediction_loss_only=prediction_loss_only,
        eval_steps=eval_steps,
    )

    """
    Then finally pass everything to the trainer
    """

    max_seq_length = tokenizer.model_max_length

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    print("Max VRAM (GB): ", max_memory)

    end = time.time()
    print("Total time (sec): ", end - start)


def main():
    parser = argparse.ArgumentParser(
        description="Simple example of a single GPU training script."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Num training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed.",
    )
    args = parser.parse_args()
    print(f"Training args: {args}")
    training_function(args)


if __name__ == "__main__":
    main()
