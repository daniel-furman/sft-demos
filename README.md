# Supervised finetuning of instruction-following LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This repo contains lightweight demos for supervised finetuning (SFT) of large language models, like MosaicML's [MPT-7B](https://huggingface.co/mosaicml/mpt-7b). In particular, we focus on short-form instruction following.

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
* See the `./inf_tests` folder for runtime tests on different models.

## Base models and datasets

In this repo, we finetune the following base models. Each is open-source and licensed for commercial use.

* [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) (Apache 2.0) 

We test the following instruction datasets. Each is open-source and licensed for commercial use.

* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (Apache 2.0)
* [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin) (Apache 2.0)

## Results

### 1. [`mpt-7b-dolphin`](https://huggingface.co/dfurman/mpt-7b-dolphin))

This short-form instruction following model was built by finetuning [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) on the first 100k rows of the [ehartford/dolphin](https://huggingface.co/datasets/ehartford/dolphin) dataset (an open-source implementation of [Microsoft's Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/)). It was trained on a single H100 (80 GB PCIe) for about 12 hours using the [Lambda Labs Platform](https://cloud.lambdalabs.com/instances).

![loss_curves](assets/jul_5_23_3_15_00_log_loss_curves_mpt-7b-dolphin.png)

Plot derived from the tfevents log at `runs/jul_5_23_3_15_00_sft-instruction-mpt-7b-dolphin` ([link](https://github.com/daniel-furman/sft-demos/tree/main/runs/jul_5_23_3_15_00_sft-instruction-mpt-7b-dolphin)).  

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
>You are a helpful assistant. Daniel is in need of a haircut. His barber works Mondays, Wednesdays, and Fridays. So, Daniel went in for a haircut on Sunday. Does this make logical sense? Let's work this out in a step by step fashion to make sure we have the right answer.\n

**mpt-7b-dolphin**:

> Step 1: Identify the days of the week when the barber works.
> 
> The barber works on Mondays, Wednesdays, and Fridays.
> 
> Step 2: Determine the day of the week on which Daniel went for a haircut.
> 
> Daniel went for a haircut on Sunday.
> 
> Step 3: Compare the day of the week on which Daniel went for a haircut to the days when the barber works.
> 
> Sunday is not one of the days when the barber works.
> 
> Step 4: Determine if the statement makes logical sense.
> 
> The statement does not make logical sense because Daniel went for a haircut on a day when the barber does not work.
> 
> Answer: No, the statement does not make logical sense. Daniel should have gone for a haircut on a day when the barber works, such as Monday, Wednesday, or Friday.

<br>

Example 3: 

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

The runtime statistics above (leftmost column) were generated with following code for each test, as per the corresponding [notebook](https://github.com/daniel-furman/sft-demos/blob/main/inf_tests/runtimes_mpt_7b_dolphin.ipynb). 
