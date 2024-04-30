# k2_training

### Launch Training
```
bash scripts/pretrain_65b.sh
```

### Converting Megatron Checkpoints to HuggingFace Format
```
python convert_ckpt_to_hf.py --load_path <megatron_ckpt_dir> --save_path <huggingface_ckpt_dir>
```