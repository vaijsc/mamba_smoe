#!/bin/bash
#SBATCH --job-name=vinai_r34
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/r34_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/r34.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=24
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
conda activate /home/anhnd81/.conda/envs/moe
cd /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES

args="
--data /lustre/scratch/client/vinai/users/anhnd81/.cache/wikitext-103/  \
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
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/r34.pt \
--wandb-flag \
--project-name hier_moe \
--job-name lb_smoe_m_r34 \
"
 
echo "Training ..."
WANDB_API_KEY="99a0a70a15a59905811d9ab32443e1a18cad8b1a" python -m torch.distributed.launch --master_port 10021 --nproc_per_node=2 --use_env train_r34.py $args

echo "Evaluation ..."
WANDB_API_KEY="99a0a70a15a59905811d9ab32443e1a18cad8b1a" python -m torch.distributed.launch --master_port 10021 --nproc_per_node=2 --use_env train_r34.py $args --resume --full-eval-mode
