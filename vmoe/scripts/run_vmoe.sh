export CUDA_VISIBLE_DEVICES=2
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="/home/ubuntu/workspace/MomentumSMoE"

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
--checkpoint /home/ubuntu/workspace/MomentumSMoE/vmoe/result/vmoe.pt
"

echo "Training ..."
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env vmoe/train.py $args

echo "Evaluation ..."
python -m torch.distributed.launch --master_port 10011 --nproc_per_node=1 --use_env vmoe/train.py $args --resume --full-eval-mode
