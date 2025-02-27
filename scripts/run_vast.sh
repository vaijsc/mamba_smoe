export CUDA_VISIBLE_DEVICES=6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

args="
--data /home/anh/wikitext-103  \
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
--checkpoint /home/anh/MomentumSMoE/result/checkpoints/r72.pt
"

echo "Training ..."
CUDA_VISIBLE_DEVICES='7' python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env train_r73.py $args #\
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

# 69909127
# 70708231
# 70603018 r24
# 70701319
# 70603015 train
# 70603402 r23

# 69810439 r39
# 70603015 r39

# check loss
# check loss + + ?
