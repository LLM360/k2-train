# Training Code for LLM360 K2-65B

This repository contains the code for training K2-65B, a 65 billion parameter large
language model from LLM360.

> [!NOTE]
> This repository is under active development. If you have suggestions or find bugs, please open a GitHub issue or reach out.

### Environment
The simplest way to launch the training should be using our [Docker Image](https://hub.docker.com/layers/zazzyyyy/project-k2/0.2/images/sha256-4c4e75f613163118a9bb27ed808c6f028c558875db9724676fa0ecee7780119e?context=explore). We will provide a more detailed writeup of the environment later.

### Launch Training
To launch training, run:
```
bash scripts/pretrain_65b.sh
```

### Converting Megatron Checkpoints to HuggingFace Format
To convert model checkpoints from Megatron to HuggingFace format, run:
```
python convert_ckpt_to_hf.py --load_path <megatron_ckpt_dir> --save_path <huggingface_ckpt_dir>
```
