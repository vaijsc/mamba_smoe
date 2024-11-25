export CUDA_VISIBLE_DEVICES=2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# mkdir -p /home/ubuntu/workspace/MomentumSMoE/result/checkpoints
#sgsgsg
args="
--data /home/ubuntu/workspace/dataset/wikitext-103 \
--base_arch transformer \
--architecture smsmsmsmsmsm \
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
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--gamma1 1.0 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.9 \
--beta2 0.999 \
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/smoe.pt
"
#--checkpoint /path/to/checkpoint/directory/smoe.pt \
#block_sz: shape of input
echo "Training ..."
#CUDA_VISIBLE_DEVICES='0,1,2,3'
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=1 --use_env train_1.py $args \
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

echo "Evaluation ..."
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=1 --use_env train_1.py $args --resume --full-eval-mode \ 
# > >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1
