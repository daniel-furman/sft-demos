# Supervised finetuning of instruction-following Large Language Models (LLMs)

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for supervised finetuning (sft) of large language models, like Meta's [llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf). In particular, we focus on tuning for short-form instruction following capabilities.

## Table of contents

1. [Instruction tuning background](https://github.com/daniel-furman/sft-demos#instruction-tuning-background)
2. [Finetuned models](https://github.com/daniel-furman/sft-demos#favorites-from-this-repo)
3. [Basic usage of models](https://github.com/daniel-furman/sft-demos#basic-usage)
4. [Base models and datasets](https://github.com/daniel-furman/sft-demos#base-models-and-datasets)


---

## Instruction-tuning background

The goal of instruction-tuning is to build LLMs that are capable of following natural language instructions to perform a wide range of tasks. The below was captured from the "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" talk by Andrej Karpathy. The key points illustrated for sft:

* Collect small but high-quality datasets in the form of prompt and ideal responses. 
* Do language modeling on this data, nothing changes algorithmically from pretraining. 
* After training we get an sft model which can be deployed as assistants (and it works to some extent).

![training_pipeline](https://raw.githubusercontent.com/daniel-furman/sft-demos/main/assets/assistant_training_pipeline.png)

For more background, see any number of excellent papers on the subject, including [Self-Instruct](https://arxiv.org/pdf/2212.10560.pdf) (2023), [Orca](https://arxiv.org/pdf/2306.02707.pdf) (2023), and [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) (2022). 

## Favorites from this repo

1. [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/llama-2-70b-dolphin-v0.1)
    *  *Note*: This model was ranked 6th on 🤗's Open LLM Leaderboard in Aug 2023
2. [dfurman/Yi-6B-Instruct-v0.1](https://huggingface.co/dfurman/Yi-6B-Instruct-v0.1) 
3. [dfurman/Mistral-7B-Instruct-v0.1](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.1) 
4. [dfurman/Falcon-180B-Instruct-v0.1](https://huggingface.co/dfurman/Falcon-180B-Instruct-v0.1) 

## Basic usage

*Note*: Use the code below to get started with the sft models herein, as ran on 1x A100.  

```python
!pip install -q -U transformers peft torch accelerate bitsandbytes einops sentencepiece

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
```

```python
peft_model_id = "dfurman/Yi-6B-instruct-v0.1"
config = PeftConfig.from_pretrained(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(
    peft_model_id,
    use_fast=True,
    trust_remote_code=True,
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    model, 
    peft_model_id
)
```

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Respond as briefly as possible."},    
    {"role": "user", "content": "Tell me a recipe for a mai tai."},
]

print("\n\n*** Prompt:")
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

print("\n\n*** Generate:")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=5,
    )

response = tokenizer.decode(
    output["sequences"][0][len(input_ids[0]):], 
    skip_special_tokens=True
)
print(response)
```

<details>

<summary>Output</summary>

**Prompt**: <|im_start|>system
You are a helpful assistant. Respond as briefly as possible.<|im_end|>
<|im_start|>user
Tell me a recipe for a mai tai.<|im_end|>
<|im_start|>assistant

**Generation**: Here's one simple version of the classic Mai Tai cocktail:

1 oz White Rum (Bacardi, Don Papa, etc.) ➕ ½ oz Coconut Cream Liqueur (Malibu or Coco Lopez)
2 tsp Simple Syrup ➕ Dash Orange Bitters
3-4 Ice Cubes

Shake all ingredients in a shaker filled with ice until well chilled and strain into an old fashioned glass over fresh crushed ice. Garnish with mint leaves if desired. Enjoy!

</details>

## Base models and datasets

We finetune off of the following base models in this repo:

* [yi](https://huggingface.co/01-ai)
* [mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* [llama-2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
* [falcon](https://huggingface.co/tiiuae/falcon-180B)
* [mpt](https://huggingface.co/mosaicml/mpt-7b)

We use the following datasets in this repo:

* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin)
* [jondurbin/airoboros-2.2.1](https://huggingface.co/datasets/jondurbin/airoboros-2.2.1)
* [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

---
