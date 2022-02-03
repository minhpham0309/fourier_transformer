## Code for Language Modeling Task in Our Paper

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Instructions

Run `sh getdata.sh` to download the data.

### Training

Run following commands to reproduce results in Table 2 and 3 in our paper.

Softmax
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir ./softmax-baseline
```

Fourier
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 203 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 16 --multi_gpu --project_name 'mgk' --seed 1111 --job_name fourier --work_dir ./fourier
```