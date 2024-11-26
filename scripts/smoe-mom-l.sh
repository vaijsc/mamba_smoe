#!/bin/bash
#SBATCH --job-name=1csmoe_l
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/1csmoe_l_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/1csmoe_l.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=sdc2-hpc-dgx-a100-016
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=24
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.AnhND81@vinai.io

eval "$(conda shell.bash hook)"
conda activate moe
cd /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE
echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo $CUDA_VISIBLE_DEVICES
# mkdir -p /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints

args="
--data /lustre/scratch/client/vinai/users/anhnd81/.cache/wikitext-103/ \
--base_arch transformer \
--architecture smsmsmsmsmsmsmsmsmsmsmsm \
--gate_name smoe \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 512 \
--nheads 8 \
--block-sz 1024 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 5000 \
--niter 80 \
--batch-sz 8 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--gamma1 0.8 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/1csmoe_l.pt \
"

# bs 24 -> 12 -> 8
echo "Training ..."
# CUDA_VISIBLE_DEVICES='0,1,2,3' 
python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args

echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='0,1,2,3' 
python -m torch.distributed.launch --master_port 10013 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode
