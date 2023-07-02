## Supervised finetuning of instruction-following LLMs

This repo includes lightweight demos for supervised finetuning (SFT) of small- to medium-sized language models, like MosaicML's [MPT-7B](https://huggingface.co/mosaicml/mpt-7b).

* The scripts here are only compatible with single NVIDIA GPU workstations, such as x1 H100 GPU with 80 GB VRAM. For multiple GPU training, see PyTorch's [training with distributed data parallelism](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)/[sharded data parallelism](https://pytorch.org/docs/stable/fsdp.html) and MosaicML's [training platform](https://www.mosaicml.com/training).

### Code assets

* See the `./sft` folder for finetuning scripts and postprocessing notebooks.
* See the `./runs` folder for the raw results from each sft experiment.
* See the `./inf_tests` folder for runtime tests on different models.

## Documentation

### SFT is the second step in a typical GPT training pipeline

Below image from "[State of GPTs](https://www.youtube.com/watch?v=bZQun8Y4L2A)" by Andrej Karpathy. 

Key points for SFT:

* Collect small but high-quality datasets in the form of "prompt" and "ideal responses". 
* Do language modeling on this data, nothing changes algorithmically from pretraining. 
* After training we get an SFT model which can be deployed as assistants (and it works to some extent).
* The scripts herein perform full-parameter sft (updates each weight in the network). Other options include parameter-efficient finetuning, see HuggingFace's [peft](https://github.com/huggingface/peft).

![training_pipeline](assets/assistant_training_pipeline.png)

### Models and datasets employed

In this repo, we finetuned small- to medium-sized LLMs on various instruction-following datasets. 

* [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) (Apache 2.0) 

Several instruction-following datasets are tested herein. Each is open-source and licensed for commercial use.

* [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) (Apache 2.0)
* [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) (MIT)

### Runs

* Include loss plots and example prompts/responses for various sft models + latency tests



