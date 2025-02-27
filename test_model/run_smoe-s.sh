export CUDA_VISIBLE_DEVICES=2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# mkdir -p /home/ubuntu/workspace/MomentumSMoE/result/checkpoints
#sgsgsg
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
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /home/ubuntu/workspace/MomentumSMoE/result/checkpoints/smoe.pt
"
#--checkpoint /path/to/checkpoint/directory/smoe.pt \
#block_sz: shape of input
echo "Training ..."
#CUDA_VISIBLE_DEVICES='0,1,2,3'
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=1 --use_env train_1.py $args \
> >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1

echo "Evaluation ..."
python -m torch.distributed.launch --master_port 10014 --nproc_per_node=1 --use_env train_1.py $args --resume --full-eval-mode \ 
> >(tee -a /home/ubuntu/workspace/MomentumSMoE/result/smoe_s.txt) 2>&1
