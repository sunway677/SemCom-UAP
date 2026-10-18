# Third-party notices

The SemCom-UAP-specific code (`channel_aware_attacker.py`, `train_semcom_uap.py`,
`eval_semcom_uap.py`, `scripts/`, and the SNR conditioning in `generator.py`) is
released under the MIT license in [`LICENSE`](LICENSE). This repository also
redistributes code
and data from the projects below, which remain under their original licenses.
Full license texts are in [`licenses/`](licenses/); the original copyright
headers are kept in the individual files.

| Component | Files in this repo | License | Copyright |
|---|---|---|---|
| [ALBEF](https://github.com/salesforce/ALBEF) | `models/model_retrieval.py`, `models/model_pretrain.py`, `models/vit.py`, `models/xbert.py`, `models/tokenization_bert.py` | BSD-3-Clause ([text](licenses/ALBEF_LICENSE.txt)) | (c) 2021, Salesforce.com, Inc. |
| [BLIP](https://github.com/salesforce/BLIP) | `models/blip_model/` | BSD-3-Clause ([text](licenses/BLIP_LICENSE.txt)) | (c) 2022, Salesforce.com, Inc. |
| [X-VLM](https://github.com/zengyan-97/X-VLM) | `models/xvlm_model/` | BSD-3-Clause ([text](licenses/X-VLM_LICENSE.txt)) | (c) 2022, ByteDance Inc. |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | `models/xvlm_model/swin_transformer.py` | MIT ([text](licenses/Swin-Transformer_LICENSE.txt)) | (c) Microsoft Corporation |
| [CLIP](https://github.com/openai/CLIP) | `models/clip_model/` | MIT ([text](licenses/CLIP_LICENSE.txt)) | (c) 2021 OpenAI |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) (BERT / RoBERTa / CLIP-ViT code adapted by ALBEF, BLIP and X-VLM) | `models/xbert.py`, `models/tokenization_bert.py` (comments translated in this repository), `models/blip_model/{med,nlvr_encoder}.py`, `models/xvlm_model/{xbert,xroberta,clip_vit,tokenization_bert,tokenization_roberta}.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) The Google AI Language Team Authors, The HuggingFace Inc. team, NVIDIA Corporation, The OpenAI Team Authors |
| [timm](https://github.com/huggingface/pytorch-image-models) (ViT code adapted by ALBEF, BLIP and X-VLM) | `models/vit.py`, `models/blip_model/vit.py`, `models/xvlm_model/vit.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) 2019 Ross Wightman |
| [latent-diffusion](https://github.com/CompVis/latent-diffusion) (via C-PGC's `attention.py`) | cross-attention blocks in `generator.py` | MIT ([text](licenses/latent-diffusion_LICENSE.txt)) | (c) 2022 Machine Vision and Learning Group, LMU Munich |
| [DETR](https://github.com/facebookresearch/detr) | `models/xvlm_model/box_ops.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) Facebook, Inc. and its affiliates |
| [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks) (Fang et al., ICCV 2025) | `attacker.py`, `utils.py`, `dataset.py`, `configs/`, the generator backbone in `generator.py`, and the base structure of the training/evaluation pipeline; `word_idx_ALBEF.npy` / `word_idx_CLIP.npy` are the vocabulary index files read by the C-PGC evaluation code (not included in the C-PGC repository) | no license file in the upstream repository | (c) the C-PGC authors |
| Flickr30K annotations (as distributed with C-PGC) | `data_annotation/flickr30k_{train,test}.json` | [Flickr30K terms of use](https://shannon.cs.illinois.edu/DenotationGraph/) (non-commercial research) | (c) the Flickr30K authors |

The Flickr30K images and the fine-tuned VLP checkpoints are not redistributed;
download them from the original sources listed in the README.
