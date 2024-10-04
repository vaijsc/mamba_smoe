mkdir -p /root/checkpoints/sst5/test/

args="
--data /root/repos/Sentiment-Analysis-SST5/semtiment_analysis/data/ \
--data_name sst5 \
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
--load_balance 0.00 \
--optim adam \
--lr 0.0001 \
--lr-warmup 1 \
--niter 5 \
--batch-sz 16 \
--batch-split 2 \
--nbatches 1000 \
--distributed \
--pretrained_weight /root/checkpoints/smoe/baseline_cpy/smoe.pt \
--checkpoint /root/checkpoints/sst5/test/smoe.pt \
"
# --wandb-flag \
# --pretrained_weight /root/checkpoints/enwik8/sym-m/smoe.pt \
# --job-name sym_smoe_sst_ft \
# --project-name neurips_momentumSMoE \

echo "Training ..."
CUDA_VISIBLE_DEVICES='4' python -m torch.distributed.launch --master_port 10043 --nproc_per_node=1 --use_env /root/repos/SMoE_finetune/finetune_train.py $args 
# python /root/repos/MomentumSMoE/finetune_train.py $args 

# echo "Evaluation ..."
# python train.py $args --full-eval-mode --batch-sz 8

# args="
# --data /root/SST-2-sentiment-analysis/data/ \
# --data_name sst2 \
# --base_arch transformer \
# --architecture sgsgsgsgsgsgsgsg \
# --gate_name smoe \
# --nlayers 8 \
# --hid-sz 352 \
# --inner-hid-sz 352 \
# --nheads 8 \
# --block-sz 512 \
# --attn-span 2048 \
# --dropout 0.1 \
# --load_balance 0.00 \
# --optim adam \
# --lr 0.0001 \
# --lr-warmup 1 \
# --niter 5 \
# --batch-sz 16 \
# --batch-split 1 \
# --nbatches 1000 \
# --pretrained_weight /root/smoe/baseline/smoe.pt \
# --checkpoint /root/checkpoints/sst2/sym-m-test/smoe.pt \
# --resume 
# "

# args="
# --data /root/SST-2-sentiment-analysis/data/ \
# --data_name sst2 \
# --base_arch transformer \
# --architecture sgsgsgsgsgsg \
# --gate_name smoe \
# --nlayers 6 \
# --hid-sz 352 \
# --inner-hid-sz 352 \
# --nheads 8 \
# --block-sz 512 \
# --attn-span 1024 \
# --dropout 0.1 \
# --load_balance 0.00 \
# --optim adam \
# --lr 0.0001 \
# --lr-warmup 1 \
# --niter 5 \
# --batch-sz 16 \
# --batch-split 1 \
# --nbatches 1000 \
# --pretrained_weight /root/smoe/baseline/smoe.pt \
# --checkpoint /root/checkpoints/sst2/sym-m-test/smoe.pt \
# --resume 
# "