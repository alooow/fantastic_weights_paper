#!/bin/bash
#SBATCH -A <account name>
#SBATCH -p <partition name>
#SBATCH --mem-per-gpu=8G
#SBATCH -N 1
#SBATCH --cpus-per-gpu=9
#SBATCH --gres=gpu:8
#SBATCH --time 2880
export IMAGENET_PYTORCH=~/ImageNet # set the path to the ImageNet here.
python multiproc.py --nnodes 1 --nproc_per_node 8 imagenet_main.py --print-freq 1000 --opt_order after --master_port $1 --seed $3 --batch-size 32 --density 0.2 --update_frequency 800 --distributed true --use_wandb true --sparse true --fix false --death $2 --growth gradient  --workers 8 --tag "ImageNet_u800_b32_$2$" --save_dir "save_b32/$2/$3/" --label-smoothing 0.1 --warmup 5







