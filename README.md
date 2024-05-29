# Training Code for LLM360 K2-65B

This repository contains the code for training K2-65B, a 65 billion parameter large
language model from LLM360.

> [!NOTE]
> This repository is under active development. If you have suggestions or find bugs, please open a GitHub issue or reach out.

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
