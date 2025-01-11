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
--checkpoint /home/anh/MomentumSMoE/result/checkpoints/smoe_m_r23_1.pt \
"
 
# --wandb-flag \
# --project-name hier_moe \
# --job-name smoe_m_r23 \
echo "Training ..."
# WANDB_API_KEY=99a0a70a15a59905811d9ab32443e1a18cad8b1a 
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10009 --nproc_per_node=2 --use_env train_r23.py $args

echo "Evaluation ..."
# WANDB_API_KEY=99a0a70a15a59905811d9ab32443e1a18cad8b1a 
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 10009 --nproc_per_node=2 --use_env train_r23.py $args --resume --full-eval-mode
