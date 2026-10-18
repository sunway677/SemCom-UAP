#!/bin/bash
#SBATCH -J eval_uap_ViT-B_32
# #SBATCH -p <partition>   # uncomment and set your cluster GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

source ~/.bashrc
conda activate semcom-uap
# repo root: script location under bash, submission directory under sbatch
if [ -f "$(dirname "$0")/../train_semcom_uap.py" ]; then cd "$(dirname "$0")/.."; else cd "$SLURM_SUBMIT_DIR"; fi
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_semcom_uap.py \
  --config ./configs/Retrieval_flickr_test.yaml \
  --source_model ViT-B/32 \
  --checkpoint ./checkpoint \
  --load_dir ./output_semcom_uap_ViTB32 \
  --epoch 99 \
  --snr_list inf 20 15 10 5 0 \
  --output result_semcom_uap_ViT-B_32
