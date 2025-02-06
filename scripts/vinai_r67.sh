#!/bin/bash
#SBATCH --job-name=r67
#SBATCH --output=/lustre/scratch/client/movian/research/users/anhnd81/workspace/MomentumSMoE/result/r67_err.txt
#SBATCH --error=/lustre/scratch/client/movian/research/users/anhnd81/workspace/MomentumSMoE/result/r67.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=24G
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/moe
cd /lustre/scratch/client/movian/research/users/anhnd81/workspace/MomentumSMoE/
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

args="
--data /lustre/scratch/client/movian/research/users/anhnd81/.cache/wikitext-103 \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name smoe \
--nlayers 6 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 1024 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 4 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/movian/research/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/r67.pt \
"
 
echo "Training ..."
# WANDB_API_KEY="99a0a70a15a59905811d9ab32443e1a18cad8b1a" 
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=2 --use_env train_r67.py $args

echo "Evaluation ..."
# WANDB_API_KEY="99a0a70a15a59905811d9ab32443e1a18cad8b1a" 
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=2 --use_env train_r67.py $args --resume --full-eval-mode
