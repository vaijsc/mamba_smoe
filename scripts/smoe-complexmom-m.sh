mkdir -p /root/checkpoints/complexSMoE/

args="
--data /root/wikitext-103/ \
--base_arch transformer \
--architecture scscscscscsc \
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
--gamma1 1.0 \
--gamma2 1.0 \
--mu 0.7 \
--beta1 0.7 \
--beta2 0.1 \
--checkpoint /root/checkpoints/complexSMoE/smoe.pt \
"
# --wandb-flag \
# --job-name complexMoM_1 \
# --project-name neurips_momentumSMoE

# echo "Training ..."
# CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env /root/repos/MomentumSMoE/train.py $args

# echo "Evaluation ..."
CUDA_VISIBLE_DEVICES='5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=3 --use_env /root/repos/MomentumSMoE/train.py $args --resume --full-eval-mode
