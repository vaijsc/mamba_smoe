mkdir -p /root/checkpoints/enwik8/sym-m/

args="
--data /root/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsg \
--gate_name smoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 90 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--checkpoint /root/checkpoints/enwik8/sym-m/smoe.pt \
--wandb-flag \
--job-name sym_smoe_enwik \
--project-name neurips_momentumSMoE \
--resume \
"
# --wandb-flag \
# --job-name sym_smoe_enwik \
# --project-name neurips_momentumSMoE \
echo "Training ..."
CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --master_port 10013 --nproc_per_node=4 --use_env /root/repos/MomentumSMoE/train.py $args

# echo "Evaluation ..."
# python train.py $args --full-eval-mode --batch-sz 8