# Supervised finetuning of instruction-following LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for supervised finetuning (sft) of large language models, like Meta's [llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf). In particular, we focus on tuning for short-form instruction following capabilities.

## Table of contents

1. [Instruction tuning background](https://github.com/daniel-furman/sft-demos#instruction-tuning-background)
2. [Code assets](https://github.com/daniel-furman/sft-demos#code-assets)
3. [Base models and datasets](https://github.com/daniel-furman/sft-demos#base-models-and-datasets)
4. [Finetuned models](https://github.com/daniel-furman/sft-demos#finetuned-models)
5. [Basic usage of peft models](https://github.com/daniel-furman/sft-demos#basic-usage-of-peft-models)

---

## Instruction tuning background

 In recent years, there has been a growing interest in building models that can follow natural language instructions to perform a wide range of tasks. These models, known as "instruction-tuned" language models, have demonstrated the ability to generalize to new tasks.
 
 The below was captured from the "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" talk by Andrej Karpathy. The key points illustrated for SFT:

* Collect small but high-quality datasets in the form of "prompt" and "ideal responses". 
* Do language modeling on this data, nothing changes algorithmically from pretraining. 
* After training we get an SFT model which can be deployed as assistants (and it works to some extent).

![training_pipeline](https://raw.githubusercontent.com/daniel-furman/sft-demos/main/assets/assistant_training_pipeline.png)

For more background, see any number of excellent papers on the subject, including [Self-Instruct](https://arxiv.org/pdf/2212.10560.pdf) (2023), [Orca](https://arxiv.org/pdf/2306.02707.pdf) (2023), and [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) (2022). 

## Code assets

* See the `./sft` folder for finetuning scripts and postprocessing notebooks.
* See the `./runs` folder for the raw results from each sft experiment.
* See the `./inf_tests` folder for runtime testing notebooks.

## Base models and datasets

In this repo, we finetune the following base models. Each is open-source and licensed for commercial use.

* [yi](https://huggingface.co/01-ai)
* [mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* [llama-2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
* [falcon](https://huggingface.co/tiiuae/falcon-180B)
* [mpt](https://huggingface.co/mosaicml/mpt-7b)

We test the following datasets. Each is open-source and licensed for commercial use.

* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin)
* [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)

## Favorite sfts

<br>

1. [dfurman/Yi-6B-instruct-v0.1](https://huggingface.co/dfurman/Yi-6B-instruct-v0.1) 
2. [dfurman/falcon-180b-instruct-peft](https://huggingface.co/dfurman/falcon-180b-instruct-peft) 
3. [dfurman/llama-2-70b-dolphin-peft](https://huggingface.co/dfurman/llama-2-70b-dolphin-peft)


## Basic usage with peft adapters

```python
!pip install -q -U huggingface_hub peft transformers torch accelerate
```

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
```

```python
peft_model_id = "dfurman/Yi-6B-instruct-v0.1"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, peft_model_id)

format_template = "You are a helpful assistant. Write a response that appropriately completes the request. {query}\n"
```

```python
query = "Tell me a recipe for vegan banana bread."
prompt = format_template.format(query=query)

print("\n\n*** Generate:")

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
    )

print(tokenizer.decode(output["sequences"][0], skip_special_tokens=True))
```
