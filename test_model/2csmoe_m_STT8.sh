#!/bin/bash
#SBATCH --job-name=STT8
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_STT8_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_STT8.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=sdc2-hpc-dgx-a100-007
#SBATCH --mem-per-gpu=50G
#SBATCH --cpus-per-gpu=24
#SBATCH --partition=applied
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
--data /lustre/scratch/client/vinai/users/anhnd81/.cache/wikitext/ \
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
--lr 0.00007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/2csmoe_STT8.pt \
"
 
# lr 0.0007
# bs 48 -> 16 -> 32
echo "Training ..."
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10002 --nproc_per_node=2 --use_env train_STT8.py $args

echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10002 --nproc_per_node=2 --use_env train_STT8.py $args --resume --full-eval-mode
