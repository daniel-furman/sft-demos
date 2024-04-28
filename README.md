# Finetuning demos for LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for finetuning of Large Language Models (LLMs), like Meta's [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B). In particular, we focus on tuning for short-form instruction following capabilities and conversational chat capabilities.

---

## 📚 Background

The goal of instruction-tuning is to train LLMs that are capable of following natural language instructions to perform a wide range of tasks. The below was captured from the "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" talk by Andrej Karpathy:

![training_pipeline](https://raw.githubusercontent.com/daniel-furman/sft-demos/main/assets/assistant_training_pipeline.png)

For more background, see any number of excellent papers on the subject, like [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) (2022). 

## 🔎 Finetuned models

See `src` for all finetuning runs. Here are some of my favorites:

1. [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1)
    *  *Note*: This model was ranked 6th on 🤗's Open LLM Leaderboard in Aug 2023
2. [dfurman/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/dfurman/Mixtral-8x7B-Instruct-v0.1)
3. [dfurman/Mistral-7B-Instruct-v0.2](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.2)

## 💻 Usage

*Note*: Use the code below to get started. Be sure to have a GPU-enabled cluster.

## 💻 Usage

<details>

<summary>Setup</summary>

```python
!pip install -qU transformers accelerate

from transformers import AutoTokenizer
import transformers
import torch

if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

model = "dfurman/Llama-3-8B-Orpo-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
)
```

</details>

### Run

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a recipe for a spicy margarita."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("***Prompt:\n", prompt)

outputs = pipeline(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print("***Generation:\n", outputs[0]["generated_text"][len(prompt):])
```

<details>

<summary>Output</summary>

```
"""***Prompt:
 <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Tell me a recipe for a spicy margarita.<|im_end|>
<|im_start|>assistant

***Generation:
 Sure! Here's a recipe for a spicy margarita:

Ingredients:

- 2 oz silver tequila
- 1 oz triple sec
- 1 oz fresh lime juice
- 1/2 oz simple syrup
- 1/2 oz fresh lemon juice
- 1/2 tsp jalapeño, sliced (adjust to taste)
- Ice cubes
- Salt for rimming the glass

Instructions:

1. Prepare the glass by running a lime wedge around the rim of the glass. Dip the rim into a shallow plate of salt to coat.
2. Combine the tequila, triple sec, lime juice, simple syrup, lemon juice, and jalapeño slices in a cocktail shaker.
3. Add ice cubes to the cocktail shaker and shake vigorously for 30 seconds to 1 minute.
4. Strain the cocktail into the prepared glass.
5. Garnish with a lime wedge and jalapeño slice.

Enjoy! This spicy margarita has a nice balance of sweetness and acidity, with a subtle heat from the jalapeño that builds gradually as you sip."""
```
</details>

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
