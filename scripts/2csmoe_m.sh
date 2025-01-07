#!/bin/bash
#SBATCH --job-name=32ft
#SBATCH --output=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/32ft_err.txt
#SBATCH --error=/lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/32ft.txt
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
export TORCH_DISTRIBUTED_DEBUG=DETAIL


args1="
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
--niter 20 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/32ft_1.pt \
"
 
# 0.0007
# bs 48 -> 16 -> 32
echo "Training ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_ft.py $args1

echo "Evaluation ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_ft.py $args1 --resume --full-eval-mode

args2="
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
--niter 20 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/32ft_2.pt \
"
 
# 0.0007
# bs 48 -> 16 -> 32
echo "Training ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_ft1.py $args2

echo "Evaluation ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_ft1.py $args2 --resume --full-eval-mode

args3="
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
--niter 20 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/32ft_3.pt \
"
 
# 0.0007
# bs 48 -> 16 -> 32
echo "Training ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_1ft.py $args3

echo "Evaluation ..."
# # CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10007 --nproc_per_node=2 --use_env train_1ft.py $args3 --resume --full-eval-mode

args4="
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
--niter 20 \
--batch-sz 32 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /lustre/scratch/client/vinai/users/anhnd81/workspace/MomentumSMoE/result/checkpoints/32ft_4.pt \
"
 
# 0.0007
# bs 48 -> 16 -> 32
echo "Training ..."
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=2 --use_env train_1ft1.py $args4

echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='0,1' 
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=2 --use_env train_1ft1.py $args4 --resume --full-eval-mode