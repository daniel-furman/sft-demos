# Finetuning demos for LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

## 📚 Intro

This repo contains demos for parameter-efficient finetuning of Large Language Models (LLMs), like Meta's [llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B). In particular, we focus on tuning for short-form instruction following and conversational chat capabilities.

---

## 🔎 Finetunes

*Note*: See `_peft` for training runs, which are organized by base model. 

* [dfurman/CalmeRys-78B-Orpo-v0.1](https://huggingface.co/dfurman/CalmeRys-78B-Orpo-v0.1)
* [dfurman/Qwen2-72B-Orpo-v0.1](https://huggingface.co/dfurman/Qwen2-72B-Orpo-v0.1)
* [dfurman/Llama-3-8B-Orpo-v0.1](https://huggingface.co/dfurman/Llama-3-8B-Orpo-v0.1)
* [dfurman/Llama-3-70B-Orpo-v0.1](https://huggingface.co/dfurman/Llama-3-70B-Orpo-v0.1)
* [dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1)
* [dfurman/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/dfurman/Mixtral-8x7B-Instruct-v0.1)
* [dfurman/Mistral-7B-Instruct-v0.2](https://huggingface.co/dfurman/Mistral-7B-Instruct-v0.2)

## 🏆 Evaluation

*Note*: See `_eval` for evaluation runs. Also see 🤗's Open LLM Leaderboard [here](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

### Open LLM Leaderboard v2

[dfurman/CalmeRys-78B-Orpo-v0.1](https://huggingface.co/dfurman/CalmeRys-78B-Orpo-v0.1)

*coming*

[dfurman/Qwen2-72B-Orpo-v0.1](https://huggingface.co/dfurman/Qwen2-72B-Orpo-v0.1)

Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_dfurman__Qwen2-72B-Orpo-v0.1)

|      Metric       |Value|
|-------------------|----:|
|Avg.               |43.32|
|IFEval (0-Shot)    |78.80|
|BBH (3-Shot)       |57.41|
|MATH Lvl 5 (4-Shot)|35.42|
|GPQA (0-shot)      |17.90|
|MuSR (0-shot)      |20.87|
|MMLU-PRO (5-shot)  |49.50|

### Open LLM Leaderboard v1

[dfurman/Llama-2-70B-Instruct-v0.1](https://huggingface.co/dfurman/Llama-2-70B-Instruct-v0.1) 

| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 65.72   |
| ARC (25-shot)         | 69.62          |
| HellaSwag (10-shot)   | 86.82    |
| MMLU (5-shot)         | 69.18         |
| TruthfulQA (0-shot)   | 57.43   |
| Winogrande (5-shot)   | 83.9   |
| GSM8K (5-shot)        | 27.37        |


## 💻 Usage

*Note*: Use the code below to get started. Be sure to have a GPU-enabled cluster.

<details>

<summary>Setup</summary>

```python
!pip install -qU transformers accelerate bitsandbytes
!huggingface-cli download dfurman/CalmeRys-78B-Orpo-v0.1
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

# # quantize if necessary
# bnb_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch_dtype,
#    bnb_4bit_use_double_quant=True,
# )

model = "dfurman/CalmeRys-78B-Orpo-v0.1"

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

</details>

### Example 1

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

```
***Generation:
|1|Calculate total sold|Add morning and afternoon sales|132|
|2|Subtract sold from total|200 - 132|68|
|3|Adjust for returns|Add returned loaves to remaining|74|
```

### Example 2

```python
question = "What's a good recipe for a spicy margarita?"

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

```
***Generation:
To make a Spicy Margarita, you'll need to incorporate a chili or pepper element into your classic margarita recipe. Here’s a simple way to do it:

### Ingredients:
- 2 oz tequila (blanco or reposado)
- 1 oz fresh lime juice
- 1/2 oz triple sec (Cointreau or Grand Marnier)
- 1/2 oz agave syrup or simple syrup
- 1-2 slices of jalapeño (or more depending on how spicy you like it)
- Salt and/or chili powder for rimming the glass
- Ice
- Lime wheel for garnish

### Instructions:
1. **Muddle Jalapeño**: In a shaker, muddle the jalapeño slices slightly. This will release the oils and heat from the peppers.
2. **Add Remaining Ingredients**: Add the tequila, lime juice, triple sec, and agave syrup or simple syrup. 
3. **Shake and Strain**: Fill the shaker with ice and shake vigorously until cold. Strain into a salt and/or chili powder rimmed glass filled with ice.
4. **Garnish and Serve**: Garnish with a lime wheel and enjoy.

If you prefer a smoother spiciness that doesn't overpower the drink, you could also consider making a jalapeño-infused tequila by leaving the jalapeño slices in the bottle of tequila for several hours to a couple of days, adjusting the time based on desired level of spiciness. Then use this infused tequila instead of regular tequila in the recipe above. 

Another variation is to use a spicy syrup. To make this, combine equal parts water and sugar with a few sliced jalapeños in a saucepan. Bring to a boil, stirring occasionally to dissolve the sugar. Reduce heat and simmer for about 5 minutes. Let cool, strain out the jalapeños, then store in a sealed container in the refrigerator until ready to use. Use this spicy syrup instead of regular syrup in the recipe. 

As always, adjust the quantity of jalapeño or the type of chili used to suit your taste. Enjoy responsibly!
```

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
