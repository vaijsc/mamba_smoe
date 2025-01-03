#!/bin/bash
#SBATCH --job-name=r13_18
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_r13_18_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/2csmoe_r13_18.txt
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
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/2csmoe_r13_18.pt \
"
 
echo "Training Hierarchical MoE idea, split two subset of experts and each choose top-2, clean data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10016 --nproc_per_node=2 --use_env train_r18.py $args

echo "Evaluation Hierarchical MoE idea, split two subset of experts and each choose top-2, clean data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10016 --nproc_per_node=2 --use_env train_r18.py $args --resume --full-eval-mode
 
echo "Training D -> 2D, N x 2D and K x 2D, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=2 --use_env train_r13.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=2 --use_env train_r13.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with RELU activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10012 --nproc_per_node=2 --use_env train_r14.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with RELU activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10012 --nproc_per_node=2 --use_env train_r14.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Sigmoid activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10013 --nproc_per_node=2 --use_env train_r15.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Sigmoid activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10013 --nproc_per_node=2 --use_env train_r15.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Tanh activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=2 --use_env train_r16.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Tanh activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=2 --use_env train_r16.py $args --resume --full-eval-mode

echo "Training D -> 2D, N x 2D and K x 2D, with Softplus activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10015 --nproc_per_node=2 --use_env train_r17.py $args

echo "Evaluation D -> 2D, N x 2D and K x 2D, with Softplus activation, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10015 --nproc_per_node=2 --use_env train_r17.py $args --resume --full-eval-mode

echo "Training Hierarchical MoE idea, split two subset of experts and each choose top-2, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10016 --nproc_per_node=2 --use_env train_r19.py $args

echo "Evaluation Hierarchical MoE idea, split two subset of experts and each choose top-2, attack data"
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10016 --nproc_per_node=2 --use_env train_r19.py $args --resume --full-eval-mode