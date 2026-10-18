#!/bin/bash
#SBATCH -J semcom_uap_ALBEF
# #SBATCH -p <partition>   # uncomment and set your cluster GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

source ~/.bashrc
conda activate semcom-uap
# repo root: script location under bash, submission directory under sbatch
if [ -f "$(dirname "$0")/../train_semcom_uap.py" ]; then cd "$(dirname "$0")/.."; else cd "$SLURM_SUBMIT_DIR"; fi
export PYTHONPATH=$PWD

python train_semcom_uap.py \
  --config ./configs/Retrieval_flickr_train.yaml \
  --seed 42 \
  --eps 12 \
  --save_dir output_semcom_uap \
  --primary_encoder ALBEF \
  --primary_ckpt ./checkpoint/ALBEF/flickr30k.pth \
  --same_aux_encoders TCL XVLM BLIP \
  --same_aux_ckpts \
    ./checkpoint/TCL/flickr30k.pth \
    ./checkpoint/XVLM/flickr30k.pth \
    ./checkpoint/BLIP/flickr30k.pth \
  --cross_aux_encoders ViT-B/32 "ViT-B/16" \
  --cross_aux_ckpts "" "" \
  --batch_size 32 \
  --epochs 100 \
  --aux_warmup_epochs 40 \
  --lambda_sur 0.65 \
  --snr_min 0 \
  --snr_max 20 \
  --margin 0.02 \
  --num_snr_strata 3
