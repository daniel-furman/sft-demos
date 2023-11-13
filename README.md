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

* See the `src/sft` folder for sft examples.

## Favorite sft llms

1. [dfurman/llama-2-70b-instruct-v0.1](https://huggingface.co/dfurman/llama-2-70b-dolphin-v0.1)
    *  *Note*: This model was ranked 6th on the Open LLM Leaderboard on Aug 10, 2023.
2. [dfurman/Yi-6B-instruct-v0.1](https://huggingface.co/dfurman/Yi-6B-instruct-v0.1) 
3. [dfurman/mistral-7b-instruct-v0.1](https://huggingface.co/dfurman/mistral-7b-instruct-v0.1) 
4. [dfurman/falcon-180b-instruct-v0.1](https://huggingface.co/dfurman/falcon-180b-instruct-v0.1) 


## Basic usage

*Note* Executed on a Google Colab notebook with 1x A100 40 GB (SXM) GPU. 

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
peft_model_id = "dfurman/mistral-7b-instruct-v0.1"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path,
    use_fast=True, 
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, peft_model_id)

format_template = "You are a helpful assistant. Write a response that appropriately completes the request. {query}\n"
```

```python
query = "Write a short email inviting my friends to a dinner party on Friday. Respond succinctly."
prompt = format_template.format(query=query)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=5,
    )

print("\n\n*** Generate:")
print(tokenizer.decode(output["sequences"][0][len(input_ids[0]):], skip_special_tokens=True))
```

<details>

<summary>Output</summary>

**Prompt**: Write a short email inviting my friends to a dinner party on Friday. Respond succinctly.

**Generation**: The invitation should be brief and to-the-point, so it's best to use simple language and avoid unnecessary details or long explanations. Here is an example of a concise invitation:

Dear Friends,

I hope you can join me for a fun evening at my place this Friday! We'll have delicious food, great conversation, and maybe even some games if we feel like it. Please RSVP by Wednesday night so I know who will be there. 

Looking forward to seeing you all soon!

Best regards,
Your Name

This message clearly communicates the essential information about the event while maintaining a friendly tone. It also includes a specific date (Friday) and timeframe (evening), as well as a clear call to action (RSVP). The closing line adds a personal touch and expresses excitement for the gathering. Overall, this invitation strikes a good balance between being informative and engaging without overwhelming the reader with too much text.

Remember, when writing emails, always keep in mind your audience and their preferences. If they prefer more detailed information or additional context, adjust accordingly. However, try not to make the invitation overly complicated or lengthy – simplicity often makes for a better experience. Happy emailing!

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
* [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

---
