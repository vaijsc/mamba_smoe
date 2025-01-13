export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

eval "$(conda shell.bash hook)"
conda activate moe
# echo "Current path is $PATH"
echo "Running"
# nvidia-smi
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

args="
--data /home/ubuntu/workspace/dataset/wikitext-103 \
--base_arch transformer \
--architecture sgsgsg \
--gate_name smoe \
--nlayers 3 \
--hid-sz 128 \
--inner-hid-sz 128 \
--nheads 8 \
--block-sz 256 \
--attn-span 256 \
--dropout 0.7 \
--load_balance 0.01 \
--optim sgd \
--lr 0.0007 \
--lr-warmup 0 \
--niter 60 \
--batch-sz 8 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/r29.pt
"

#--checkpoint /path/to/checkpoint/directory/smoe.pt \
#block_sz: shape of input
echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env train_r30.py $args #\
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

# echo "Evaluation ..."
# python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env train.py $args --resume --full-eval-mode #\ 
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

# 69909127
# 70708231
# 70603018 r24

# 70603015 train
# 70603402 r23
# 

# check loss
# check loss + + ?
