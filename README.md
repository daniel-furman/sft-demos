# Supervised finetuning of instruction-following LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains demos for supervised finetuning (sft) of large language models, like MosaicML's [mpt](https://huggingface.co/mosaicml/mpt-7b) and Meta's [llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf). In particular, we focus on short-form instruction following.

## Instruction tuning background

 In recent years, there has been a growing interest in building models that can follow natural language instructions to perform a wide range of tasks. These models, known as "instruction-tuned" language models, have demonstrated the ability to generalize to new tasks.
 
 The below was captured from the "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" talk by Andrej Karpathy. The key points illustrated for SFT:

* Collect small but high-quality datasets in the form of "prompt" and "ideal responses". 
* Do language modeling on this data, nothing changes algorithmically from pretraining. 
* After training we get an SFT model which can be deployed as assistants (and it works to some extent).

![training_pipeline](assets/assistant_training_pipeline.png)

For more background, see any number of excellent papers on the subject, including [Self-Instruct](https://arxiv.org/pdf/2212.10560.pdf) (2023), [Orca](https://arxiv.org/pdf/2306.02707.pdf) (2023), and [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) (2022). 

## Code assets

* See the `./sft` folder for finetuning scripts and postprocessing notebooks.
* See the `./runs` folder for the raw results from each sft experiment.
* See the `./inf_tests` folder for runtime testing notebooks.

## Base models and datasets

In this repo, we finetune the following base models. Each is open-source and licensed for commercial use.

* [mpt](https://huggingface.co/mosaicml/mpt-7b)
* [llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)

We test the following datasets. Each is open-source and licensed for commercial use.

* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin)

## Finetuned models

### 1. [`llama-2-70b-dolphin-peft`](https://huggingface.co/dfurman/llama-2-70b-dolphin-peft)

This instruction model was built via parameter-efficient QLoRA finetuning of [llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf) on the first 25k rows of [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin) (an open-source implementation of [Microsoft's Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/)). Finetuning was executed on a single H100 (80 GB PCIe) for roughly 17 hours on the [Lambda Labs](https://cloud.lambdalabs.com/instances) platform.

* Model license: Llama 2 Community License Agreement
* Basic usage: [notebook](https://huggingface.co/dfurman/llama-2-70b-dolphin-peft/blob/main/assets/basic_inference_llama_2_70b_dolphin.ipynb)
* Finetuning code: [script](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/dolphin/sft-llama-2-70b-dolphin-peft.py)

![loss_curves](assets/jul_24_23_1_14_00_log_loss_curves_llama-2-70b-dolphin.png)

The above loss curve was generated from the run's private wandb.ai log.  

### Example prompts and responses

Example 1: 

**User**:
> You are a helpful assistant. Write me a numbered list of things to do in New York City.\n

**llama-2-70b-dolphin-peft**:
> 1. Visit the Statue of Liberty: Take a ferry to Liberty Island and see the iconic symbol of freedom and democracy.
> 2. Explore Central Park: Enjoy the lush greenery, take a stroll, or rent a bike to explore the park's many attractions, such as the Central Park Zoo and Strawberry Fields.
> 3. Visit the Empire State Building: Take an elevator to the observation deck for breathtaking views of the city.
> 4. Walk the Brooklyn Bridge: Cross the historic bridge and enjoy the stunning views of the Manhattan skyline.
> 5. Experience Times Square: Take in the bright lights, billboards, and bustling energy of this iconic intersection.
> 6. Visit the Metropolitan Museum of Art: Explore the vast collection of art and artifacts from around the world.
> 7. Take a food tour: Sample the diverse cuisine of New York City, from pizza to bagels to ethnic specialties.
> 8. Attend a Broadway show: Enjoy a world-class performance in the heart of the Theater District.

<br>

Example 2: 

**User**:

> You are a helpful assistant. Write a short email inviting my friends to a dinner party on Friday. Respond succinctly.\n

**llama-2-70b-dolphin-peft**:

> Subject: Dinner Party Invitation - Friday, 7 PM
> 
> Dear Friends,
> 
> I hope this email finds you well! I'm excited to invite you to a dinner party at my place this Friday, at 7 PM. We'll have a delicious meal, great company, and lots of fun. Please let me know if you can make it, so I can plan accordingly.
> 
> Looking forward to seeing you all!
> 
> Kind regards,
> Your Name

> P.S. Please let me know if you have any dietary restrictions or allergies.

<br>

Example 3: 

**User**:

> You are a helpful assistant. Tell me a recipe for vegan banana bread.\n

**llama-2-70b-dolphin-peft**:

> Sure! Here's a delicious and easy vegan banana bread recipe:
> 
> Ingredients:
> - 2 cups all-purpose flour
> - 1/2 cup sugar
> - 1/2 cup vegan butter (such as Earth Balance)
> - 1/2 cup vegan milk (such as almond milk)
> - 1/2 cup unsweetened applesauce
> - 1/2 cup mashed ripe bananas (about 2 medium bananas)
> - 1 teaspoon baking soda
> - 1/2 teaspoon salt
> - 1/2 teaspoon ground cinnamon
> - 1/2 teaspoon ground nutmeg
> - 1/2 teaspoon ground cloves
> - 1/2 cup chopped walnuts (optional)
> 
> Instructions:
> 1. Preheat the oven to 350°F (175°C). Grease a 9x5-inch loaf pan with vegan butter or cooking spray.
> 2. In a large bowl, mix together the flour, sugar, vegan butter, vegan milk, applesauce, bananas, baking soda, salt, cinnamon, nutmeg, and cloves. Stir until well combined.
> 3. Fold in the chopped walnuts, if using.
> 4. Pour the batter into the prepared loaf pan.
> 5. Bake for 50-60 minutes, or until a toothpick inserted into the center of the bread comes out clean.
> 6. Let the bread cool in the pan for 10 minutes before transferring it to a wire rack to cool completely.
> 7. Slice and enjoy!
> 
> Note: You can also add chocolate chips, dried fruit, or other mix-ins to the batter for extra flavor and texture. Enjoy your vegan banana bread!

<br>

### Runtime tests

| runtime / 50 tokens (sec) | GPU             | attn | torch dtype | VRAM (GB) |
|:-----------------------------:|:----------------------:|:---------------------:|:-------------:|:-----------------------:|
| 4.50                        | 1x H100 (80 GB PCIe)  | torch               | nf4    | 39                    |

The above runtime stats were generated from this [notebook](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/dolphin/postprocessing-llama-2-70b-dolphin-peft.ipynb). 

<br>

### 2. [`llama-2-13b-guanaco-peft`](https://huggingface.co/dfurman/llama-2-13b-guanaco-peft)

This chatbot model was built via parameter-efficient QLoRA finetuning of [llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf) on all 9.85k rows of [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (a subset of [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) containing the highest-rated conversation paths). Finetuning was executed on a single A6000 (48 GB) for roughly 3.7 hours on the [Lambda Labs](https://cloud.lambdalabs.com/instances) platform.

* Model license: Llama 2 Community License Agreement
* Basic usage: [notebook](https://huggingface.co/dfurman/llama-2-13b-guanaco-peft/blob/main/assets/basic_inference_llama_2_13b_guanaco.ipynb)
* Finetuning code: [notebook](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/guanaco/sft-llama-2-13b-guanaco-peft.ipynb)

![loss_curves](assets/jul_22_23_3_15_00_log_loss_curves_llama-2-13b-guanaco.png)

The above loss curve was generated from the run's private wandb.ai log.  

### Example prompts and responses

Example 1: 

**User**:
>\### Human: Write me a numbered list of things to do in New York City.### Assistant: 

**llama-2-13b-guanaco-peft**:

> 1. Visit the Statue of Liberty and Ellis Island
> 2. Take a stroll through Central Park
> 3. Visit the Empire State Building
> 4. See a Broadway show
> 5. Explore the Metropolitan Museum of Art
> 6. Visit the 9/11 Memorial and Museum
> 7. Take a ride on the Staten Island Ferry
> 8. Shop on Fifth Avenue
> 9. Eat a slice of pizza at a local pizzeria
> 10. Visit the High Line

<br>

Example 2: 

**User**:
>\### Human: Write a short email inviting my friends to a dinner party on Friday. Respond succinctly.### Assistant:

**llama-2-13b-guanaco-peft**:

> Subject: Dinner party on Friday
> 
> Hey guys,
> 
> I'm hosting a dinner party on Friday at my place. It's going to be a small gathering with just a few of us. I'll be cooking some delicious food, so I hope you can make it.
> 
> Let me know if you can come and what you'd like to bring. I'll send out the address and more details later.
> 
> See you soon!
> 
> [Your name]

<br>

### Runtime tests

| runtime / 50 tokens (sec) | GPU             | attn | torch dtype | VRAM (GB) |
|:-----------------------------:|:----------------------:|:---------------------:|:-------------:|:-----------------------:|
| 2.93                        | 1x A100 (40 GB SXM)  | torch               | bfloat16    | 25                    |
| 3.24                        | 1x A6000 (48 GB)  | torch               | bfloat16    | 25                    |

The above runtime stats were generated from this [notebook](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/llama-2/guanaco/postprocessing-llama-2-13b-guanaco-peft.ipynb). 

<br>

### 3. [`mpt-7b-dolphin`](https://huggingface.co/dfurman/mpt-7b-dolphin)

This instruction model was built via full parameter finetuning of [mpt-7b](https://huggingface.co/mosaicml/mpt-7b) on the first 100k rows of [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin) (an open-source implementation of [Microsoft's Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/)). Finetuning was executed on a single H100 (80 GB PCIe) for roughly 12 hours on the [Lambda Labs](https://cloud.lambdalabs.com/instances) platform.

* Model license: Apache 2.0
* Basic usage: [notebook](https://huggingface.co/dfurman/mpt-7b-dolphin/blob/main/assets/basic_inference_mpt-7b-dolphin.ipynb)
* Finetuning code: [script](https://github.com/daniel-furman/sft-demos/blob/main/src/sft/one_gpu/mpt/dolphin/sft-mpt-7b-dolphin.py)

![loss_curves](assets/jul_5_23_3_15_00_log_loss_curves_mpt-7b-dolphin.png)

The above loss curve was generated from the tfevents log at [link](https://github.com/daniel-furman/sft-demos/tree/main/runs/jul_5_23_3_15_00_sft-mpt-7b-dolphin).  

### Example prompts and responses

Example 1: 

**User**:
>You are a helpful assistant. Write me a numbered list of things to do in New York City.\n

**mpt-7b-dolphin**:

> 1. Visit the Statue of Liberty: The Statue of Liberty is a symbol of freedom and democracy, and it is a must-see attraction in New York City.
> 2. Take a walk in Central Park: Central Park is a massive green space in the heart of Manhattan, and it is a great place to relax, enjoy nature, and people-watch.
> 3. Visit the Metropolitan Museum of Art: The Met is one of the world's largest and most renowned art museums, housing an extensive collection of art from around the world.
> 4. Take a stroll through Times Square: Times Square is the heart of the city, and it is a must-see for its iconic billboards, bright lights, and bustling energy.
> 5. Visit the Empire State Building: The Empire State Building is a famous skyscraper in Midtown Manhattan, offering panoramic views of the city from its observation deck.
> 6. Explore the Museum of Modern Art (MoMA): MoMA is a renowned art museum in Midtown Manhattan, showcasing modern and contemporary art from around the world.
> 7. Visit the 9/11 Memorial and Museum: The 9/11 Memorial and Museum is a somber but important attraction in Lower Manhattan, commemorating the victims of the September 11th attacks.
> 8. Take a ferry ride to the Statue of Liberty: A ferry ride to the Statue of Liberty is a great way to see the city from the water

<br>

Example 2: 

**User**:
>You are a helpful assistant. Write a short email inviting my friends to a dinner party on Friday. Respond succinctly.\n

**mpt-7b-dolphin**:

> Subject: Invitation to Dinner Party on Friday
> 
> Dear Friend,
> 
> I hope this email finds you well. I am writing to invite you to a dinner party that I am hosting on Friday. The party will be a casual gathering, and I would love to have you join us.
> 
> The dinner will be held at my home, and the time is from 7:00 pm to 10:00 pm. The address is 123 Main Street, San Francisco, CA 94122.
> 
> The dinner will be a potluck, so please bring a dish to share. I will provide the main course, so you can focus on bringing a side dish, dessert, or appetizer.
> 
> Please RSVP by replying to this email or by calling me at 555-555-5555. I look forward to seeing you on Friday!
> 
> Yours,
> 
> Your Friendly Assistant

<br>

### Runtime tests

| runtime / 50 tokens (sec) | GPU             | attn | torch dtype | VRAM (GB) |
|:-----------------------------:|:----------------------:|:---------------------:|:-------------:|:-----------------------:|
| 0.61                        | 1x H100 (80 GB PCIe) | triton              | bfloat16    | 12                    |
| 0.67                        | 1x H100 (80 GB PCIe) | torch               | bfloat16    | 12                    |
| 1.17                        | 1x A100 (40 GB SXM)  | triton              | bfloat16    | 13                    |
| 1.36                        | 1x A100 (40 GB SXM)  | torch               | bfloat16    | 13                    |
| 2.25                        | 1x V100 (16 GB SXM)  | torch               | float16     | 13                    |
| 3.75                        | 1x V100 (16 GB SXM)  | torch               | fp4         | 4                     |
| 4.84                        | 1x Tesla T4 (15 GB)  | torch               | float16     | 13                    |
| 8.77                        | 1x Tesla T4 (15 GB)  | torch               | fp4         | 4                     |

The above runtime stats were generated from this [notebook](https://github.com/daniel-furman/sft-demos/blob/main/inf_tests/runtimes_mpt_7b_dolphin.ipynb). 

<br>