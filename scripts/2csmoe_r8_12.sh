#!/bin/bash
#SBATCH --job-name=r8-12
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_r8_12_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_r8_12.txt
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
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/2csmoe_r812.pt \
"
 
echo "Training D -> 2D, N x 2D and K x 2D"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10030 --nproc_per_node=2 --use_env train_r8.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10030 --nproc_per_node=2 --use_env train_r8.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with RELU activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10031 --nproc_per_node=2 --use_env train_r9.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with RELU activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10031 --nproc_per_node=2 --use_env train_r9.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Sigmoid activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10032 --nproc_per_node=2 --use_env train_r10.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Sigmoid activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10032 --nproc_per_node=2 --use_env train_r10.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Tanh activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10033 --nproc_per_node=2 --use_env train_r11.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Tanh activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10033 --nproc_per_node=2 --use_env train_r11.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Softplus activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10034 --nproc_per_node=2 --use_env train_r12.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Softplus activation"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10034 --nproc_per_node=2 --use_env train_r12.py $args --resume --full-eval-mode
