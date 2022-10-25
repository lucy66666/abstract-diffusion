## A Visual Tour of Current Challenges in Multimodal Language Models

This codebase contains PyTorch implementation of the paper:

> A Visual Tour Of Current Challenges In Multimodal Language Models.
> Shashank Sonkar, Naiming Liu, and Richard G. Baraniuk.
> [[Paper]](https://arxiv.org/abs/2210.12565)


### Running the model
```
mkdir images
python stable_diffusion.py --input_path prompts/pronouns_prompts.txt --output_path images/pronouns
```

### Requirements
Stable diffusion model used: https://huggingface.co/CompVis/stable-diffusion-v1-4

Commit id of the model: https://huggingface.co/CompVis/stable-diffusion-v1-4/commit/f15bc7606314c6fa957b4267bee417ee866c0b84
