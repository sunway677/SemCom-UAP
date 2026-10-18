#!/bin/bash
#SBATCH -J semcom_uap_ViTB16
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_semcom_uap.py \
  --primary_encoder ViT-B/16 \
  --primary_ckpt "" \
  --same_aux_encoders ViT-B/32 \
  --same_aux_ckpts "" \
  --cross_aux_encoders ALBEF TCL BLIP XVLM \
  --cross_aux_ckpts \
      ./checkpoint/ALBEF/flickr30k.pth \
      ./checkpoint/TCL/flickr30k.pth \
      ./checkpoint/BLIP/flickr30k.pth \
      ./checkpoint/XVLM/flickr30k.pth \
  --save_dir output_semcom_uap_ViTB16 \
  --batch_size 64 --epochs 100 --aux_warmup_epochs 40 \
  --snr_min 0 --snr_max 20 \
  --margin 0.02 --num_snr_strata 3
