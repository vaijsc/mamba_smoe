args="
--data /home/anh/wikitext-103  \
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
--checkpoint /home/anh/MomentumSMoE/result/checkpoints/lb_smoe_m.pt \
"
 
echo "Training ..."
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10017 --nproc_per_node=2 --use_env train.py $args

echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10017 --nproc_per_node=2 --use_env train.py $args --resume --full-eval-mode
