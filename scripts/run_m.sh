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
--niter 3 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/smoe.pt \
"
 
# # bs 48 -> 16 -> 32
# echo "Training ..."
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10007 --nproc_per_node=4 --use_env train.py $args \
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_test.txt) 2>&1
echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port 10007 --nproc_per_node=4 --use_env train.py $args  --full-eval-mode \
# --resume --full-eval-mode
> >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_test.txt) 2>&1