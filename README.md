# Finetuning demos for LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for finetuning of Large Language Models (LLMs), like Meta's [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B). In particular, we focus on tuning for short-form instruction following capabilities and conversational chat capabilities.

---

## 🔎 Finetuned models

See `peft` for finetuning runs. 

* [dfurman/Qwen2-72B-Orpo-v0.1](https://huggingface.co/dfurman/Qwen2-72B-Orpo-v0.1)
* [dfurman/Llama-3-8B-Orpo-v0.1](https://huggingface.co/dfurman/Llama-3-8B-Orpo-v0.1)
* [dfurman/Llama-3-70B-Orpo-v0.1](https://huggingface.co/dfurman/Llama-3-70B-Orpo-v0.1)
* [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1)
    *  *Note*: This model was ranked 6th on 🤗's Open LLM Leaderboard in Aug 2023
* [dfurman/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/dfurman/Mixtral-8x7B-Instruct-v0.1)
* [dfurman/Mistral-7B-Instruct-v0.2](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.2)

## 💻 Usage

*Note*: Use the code below to get started. Be sure to have a GPU-enabled cluster.

### Setup

```python
!pip install -qU transformers accelerate bitsandbytes
!huggingface-cli download dfurman/Qwen2-72B-Orpo-v0.1
```

```python
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch


if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# quantize if necessary
# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch_dtype,
#    bnb_4bit_use_double_quant=True,
# )

model = "dfurman/Qwen2-72B-Orpo-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch_dtype,
        # "quantization_config": bnb_config,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
)
```

### Run

```python
question = """The bakers at the Beverly Hills Bakery baked 200 loaves of bread on Monday morning. 
They sold 93 loaves in the morning and 39 loaves in the afternoon. 
A grocery store then returned 6 unsold loaves back to the bakery. 
How many loaves of bread did the bakery have left?
Respond as succinctly as possible. Format the response as a completion of this table:
|step|subquestion|procedure|result|
|:---|:----------|:--------|:-----:|"""


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": question},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print("***Prompt:\n", prompt)

outputs = pipeline(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print("***Generation:")
print(outputs[0]["generated_text"][len(prompt):])
```

### Result

```
***Generation:
|1|Initial loaves|Start with total loaves|200|
|2|Sold in morning|Subtract morning sales|200 - 93 = 107|
|3|Sold in afternoon|Subtract afternoon sales|107 - 39 = 68|
|4|Returned loaves|Add returned loaves|68 + 6 = 74|
```


## 🏆 Evaluation

See `eval` for evaluation runs. 

* [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1) 

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

Base models:

* [qwen2](https://huggingface.co/Qwen/Qwen2-72B-Instruct)
* [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [phi-2](https://huggingface.co/microsoft/phi-2)
* [mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
* [mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
* [llama-2](https://huggingface.co/meta-llama/Llama-2-70b-hf)
* [falcon](https://huggingface.co/tiiuae/falcon-180B)

Datasets:

* [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin)
* [jondurbin/airoboros-2.2.1](https://huggingface.co/datasets/jondurbin/airoboros-2.2.1)
* [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

Compute providers:

* [RunPod](https://www.runpod.io/)
* [Lambda Labs](https://lambdalabs.com/)
* [Google Colab](https://colab.google/)

---
