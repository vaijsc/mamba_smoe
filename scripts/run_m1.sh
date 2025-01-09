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
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/smoe_ft3.pt \
"
 
echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10008 --nproc_per_node=4 --use_env train_ft3.py $args \
> >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_ft3.txt) 2>&1

# echo "Evaluation ..."
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10008 --nproc_per_node=4 --use_env train_ft3.py $args --resume --full-eval-mode # \
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_ft1.txt) 2>&1