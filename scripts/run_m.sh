export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

eval "$(conda shell.bash hook)"
conda activate moe
# echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

args="
--data /home/ubuntu/workspace/dataset/wikitext-103  \
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
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/mamba_smoe_4.pt \
"
 
#--checkpoint /path/to/checkpoint/directory/smoe.pt \
#block_sz: shape of input
echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env train_r22.py $args #\
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

# echo "Evaluation ..."
# python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode #\ 
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

# 69909127
# 70708231