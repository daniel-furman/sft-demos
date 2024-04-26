# Finetuning demos for LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for finetuning of Large Language Models (LLMs), like Meta's [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B). In particular, we focus on tuning for short-form instruction following capabilities and conversational chat capabilities.

---

## 📚 Background

The goal of instruction-tuning is to build LLMs that are capable of following natural language instructions to perform a wide range of tasks. The below was captured from the "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" talk by Andrej Karpathy. The key points illustrated for sft:

* Collect small but high-quality datasets in the form of prompt and ideal responses. 
* Do language modeling on this data, nothing changes algorithmically from pretraining. 
* After training we get an sft model which can be deployed as assistants (and it works to some extent).

![training_pipeline](https://raw.githubusercontent.com/daniel-furman/sft-demos/main/assets/assistant_training_pipeline.png)

For more background, see any number of excellent papers on the subject, including [Self-Instruct](https://arxiv.org/pdf/2212.10560.pdf) (2023), [Orca](https://arxiv.org/pdf/2306.02707.pdf) (2023), and [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) (2022). 

## 🔎 Finetuned models

See `src` for all finetuning runs. Here are some of my favorites:

1. [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1)
    *  *Note*: This model was ranked 6th on 🤗's Open LLM Leaderboard in Aug 2023
2. [dfurman/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/dfurman/Mixtral-8x7B-Instruct-v0.1)
3. [dfurman/Mistral-7B-Instruct-v0.2](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.2)

## 💻 Usage

*Note*: Use the code below to get started with our sft models, as ran on 1x A100 (40 GB SXM).

**dfurman/Mixtral-8x7B-Instruct-v0.1**

<details>

<summary>Setup</summary>

```python
!pip install -q -U transformers peft torch accelerate einops sentencepiece bitsandbytes
```

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
```

```python
peft_model_id = "dfurman/Mixtral-8x7B-Instruct-v0.1"
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
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    model, 
    peft_model_id
)
```

</details>


```python
messages = [
    {"role": "user", "content": "Tell me a recipe for a mai tai."},
]

print("\n\n*** Prompt:")
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
)
print(tokenizer.decode(input_ids[0]))

print("\n\n*** Generate:")
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        input_ids=input_ids.to("cuda"),
        max_new_tokens=1024,
        return_dict_in_generate=True,
    )

response = tokenizer.decode(
    output["sequences"][0][len(input_ids[0]):], 
    skip_special_tokens=True
)
print(response)
```

**Outputs**

```python
"""
*** Prompt:
<s> [INST] Tell me a recipe for a mai tai. [/INST] 

*** Generate:
1.5 oz light rum
2 oz dark rum
1 oz lime juice
0.5 oz orange curaçao
0.5 oz orgeat syrup

In a shaker filled with ice, combine the light rum, dark rum, lime juice, orange curaçao, and orgeat syrup. Shake well.

Strain the mixture into a chilled glass filled with fresh ice.

Garnish with a lime wedge and a cherry.
"""
```

## 🏆 Evaluation

See `src/eval` for all evaluation runs. 

We evaluate models on 6 key benchmarks using Eleuther.AI's Language Model Evaluation Harness.

1. [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1) 

* Precision: fp16

| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 65.72   |
| ARC (25-shot)         | 69.62          |
| HellaSwag (10-shot)   | 86.82    |
| MMLU (5-shot)         | 69.18         |
| TruthfulQA (0-shot)   | 57.43   |
| Winogrande (5-shot)   | 83.9   |
| GSM8K (5-shot)        | 27.37        |

## 🤝 References

We finetune off of the following base models:

* [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [phi-2](https://huggingface.co/microsoft/phi-2)
* [mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
* [mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* [llama-2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
* [falcon](https://huggingface.co/tiiuae/falcon-180B)

We use the following datasets:

* [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin)
* [jondurbin/airoboros-2.2.1](https://huggingface.co/datasets/jondurbin/airoboros-2.2.1)
* [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

We use the following compute providers:

* [RunPod](https://www.runpod.io/)
* [Lambda Labs](https://lambdalabs.com/)
* [Google Colab](https://colab.google/)

---
