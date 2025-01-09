args="
--data /home/ubuntu/workspace/dataset/wikitext-103 \
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
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/hier_tok1.pt \
"
 
echo "Training ..."
CUDA_VISIBLE_DEVICES='2,3,4,5' python -m torch.distributed.launch --master_port 10007 --nproc_per_node=4 --use_env train_ft5.py $args #\
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_ft.txt) 2>&1

# echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='2,3,4,5' python -m torch.distributed.launch --master_port 10007 --nproc_per_node=4 --use_env train_ft5.py $args  --full-eval-mode \
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_ft.txt) 2>&1

# 23895264
# 25384224
# 23895276
# 23895372
# 23895276

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10007 --nproc_per_node=0 --use_env train_ft1.py $args \
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_ft3.txt) 2>&1