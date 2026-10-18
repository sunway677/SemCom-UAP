# SemCom-UAP

Official PyTorch implementation of **"SemCom-UAP: Universal Adversarial Perturbations
Against Cross-Modal Retrieval in Semantic Communications"**, published at the
*IEEE Conference on Communications and Network Security (CNS), 2026*.

Ziyu Wang, Yiming Du, Jian Li, Rui Ning, Shuai Hao, Daniel Takabi, Lusi Li

SemCom-UAP is a channel-aware universal adversarial perturbation (UAP) framework that
attacks cross-modal (image-to-text / text-to-image) retrieval in vision-language
pre-training (VLP)-based semantic communication systems. Unlike prior channel-blind UAP
attacks, it conditions perturbation generation on the wireless channel SNR and trains
with a channel-robust adversarial optimization strategy combining:

- adversarial cross-modal contrastive learning on AWGN-corrupted embeddings,
- adaptive multi-surrogate training via a UCB bandit, and
- stratified SNR sampling across the operating channel range.

This repository builds on the [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks)
codebase (*"One Perturbation is Enough"*, Fang et al., ICCV 2025) for the VLP model
wrappers and data loading, and adds the SemCom-UAP generator, the channel-aware
attackers, and the training/evaluation pipeline used for the results in the paper.

## Repository layout

```
SemCom-UAP/
├── generator.py                # SNR-aware conditional generator + shared SNREncoder
├── channel_aware_attacker.py   # AWGN channel, adversarial cross-modal contrastive
│                                 loss, image/text UAP attackers
├── train_semcom_uap.py         # Training: stratified SNR sampling + UCB adaptive
│                                 surrogate selection
├── eval_semcom_uap.py          # Evaluation: Recall@1, ASR@K, Pure UAP ASR@K
├── attacker.py, utils.py, dataset.py   # Shared utilities from C-PGC (model loading,
│                                 dataset, text token substitution)
├── models/                     # VLP backbone wrappers: ALBEF, TCL, BLIP, X-VLM, CLIP
├── configs/                    # Dataset / model config files
├── data_annotation/            # Flickr30K train/test annotation JSONs
├── word_idx_ALBEF.npy, word_idx_CLIP.npy   # Vocabularies for the adversarial-word
│                                 lookup (read by the C-PGC evaluation code)
├── scripts/
│   ├── generate_word_embeddings.py       # Builds the word-embedding caches
│   ├── train_{ALBEF,TCL,BLIP,XVLM,ViTB16,ViTB32}.sh
│   └── eval_{ALBEF,TCL,BLIP,XVLM,ViT-B_16,ViT-B_32}.sh
├── logs/                       # SLURM stdout/stderr
├── LICENSE                     # MIT (SemCom-UAP code)
├── THIRD_PARTY_NOTICES.md      # Licenses of the vendored third-party code
└── licenses/
```

## Notation: paper symbols to command-line flags

| Paper | Flag | Default | Where |
|---|---|---|---|
| image budget `eps_x` | `--eps` (in 1/255) | `12` | Sec. IV-A |
| text budget `eps_t` (`K_p`) | `--text_budget` | `1` | Sec. IV-A |
| SNR range `[gamma_min, gamma_max]` | `--snr_min`, `--snr_max` | `0`, `20` dB | Sec. IV-H |
| SNR strata `M` | `--num_snr_strata` | `3` | Eq. (20) |
| temperature `tau` | `--temperature`, `--txt_temperature` | `0.07` | Eq. (12) |
| margin `m` | `--margin` | `0.02` | Eq. (12) |
| generator lr `eta_g` | `--lr` | `2e-4` | Sec. IV-H |
| latent lr scale `alpha` | `--z_lr_scale` | `0.05` | Sec. IV-H |
| warm-up `E_warm` | `--aux_warmup_epochs` | `40` | Eq. (22) |
| surrogate scale `lambda_sur` | `--lambda_sur` | `0.65` | Eq. (19) |
| exploration `c_ucb` | `--aux_ucb_exploration` | `1.0` | Eq. (18) |
| reward momentum `eta` | `--aux_ucb_momentum` | `0.1` | Eq. (17) |
| selection floor `rho_min` | `--aux_min_select_ratio` | `0.02` | Sec. IV-G |

The former flag names `--word_num`, `--k_samples` and `--lambda_aux_init` are still
accepted as aliases. The training scripts use `--batch_size 32` for the BERT-based
primary surrogates (ALBEF, TCL, BLIP, X-VLM) and `64` for the CLIP primaries.

## Setup

```bash
conda env create -f environment.yml
conda activate semcom-uap
```

`environment.yml` installs `requirements.txt`: the package versions of the C-PGC
environment plus the extra packages this code imports. The experiments in the paper
were run with Python 3.10, PyTorch 2.5.1, timm 1.0.24 and transformers 4.x.
transformers >= 5 is not supported by the vendored ALBEF/BLIP/X-VLM text encoders.
Training and evaluation require a CUDA GPU; `bert-base-uncased` (tokenizer/config) is
downloaded from the HuggingFace Hub on first use.

### Datasets and checkpoints

1. Download the [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) images and
   set `image_root` in `configs/Retrieval_flickr_train.yaml` and
   `configs/Retrieval_flickr_test.yaml` to the directory that **contains**
   `flickr30k-images/` (the annotation entries are `flickr30k-images/<id>.jpg`; the
   default `''` means `./flickr30k-images/`). The annotation files are included under
   `data_annotation/`.
2. Download the Flickr30K-finetuned retrieval checkpoints of
   [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL),
   [BLIP](https://github.com/salesforce/BLIP) (ViT-B) and
   [X-VLM](https://github.com/zengyan-97/X-VLM) (Swin-B, 384px), and save them as:
   ```
   checkpoint/
   ├── ALBEF/flickr30k.pth
   ├── TCL/flickr30k.pth
   ├── BLIP/flickr30k.pth
   └── XVLM/flickr30k.pth
   ```
   The CLIP ViT-B/16 and ViT-B/32 weights are downloaded automatically by
   `models/clip_model` on first use.
3. Build the word-embedding caches (requires `checkpoint/ALBEF/flickr30k.pth`):
   ```bash
   python scripts/generate_word_embeddings.py --all
   ```
   This writes `word_embedding_{ALBEF,TCL,XVLM,BLIP}.pth` (the fine-tuned ALBEF
   text-encoder word-embedding table, shared by the BERT-based models),
   `word_embedding_CLIP.pth` (the CLIP ViT-B/16 token-embedding table) and
   `word_idx_{TCL,XVLM,BLIP}.npy` (aliases of the shipped `word_idx_ALBEF.npy`).
   Training uses the embedding norms to clamp the text perturbation; evaluation uses the
   tables to map the text UAP to its nearest vocabulary word.

## Training

Each script trains the generators against one primary surrogate and uses the other five
backbones as UCB-selected auxiliary surrogates (same-system and cross-system). Run a
script directly, or submit it with SLURM from the repository root (the `#SBATCH -p`
partition line is commented out; set it for your cluster):

```bash
bash scripts/train_ALBEF.sh      # or: sbatch scripts/train_ALBEF.sh
```

Each iteration takes up to three separate optimizer steps:

1. **Image branch** (primary surrogate): `U(-eps, eps)` noise is added to the clean
   images, `M` SNR values are drawn by stratified sampling, and for each SNR the image
   generator produces one perturbation per sample, conditioned on the image's
   first-caption feature (from the feature bank) and the SNR embedding. The adversarial
   embeddings are corrupted by AWGN, the adversarial cross-modal contrastive (ACMC) loss
   is averaged over the strata, and the image generator, its latent code and the SNR
   encoder are updated.
2. **Text branch** (primary surrogate): `M` fresh stratified SNR draws; for each, the
   text generator (conditioned on the per-image feature and the SNR embedding) produces
   a word-embedding perturbation that is substituted at the most important token
   position of every caption, with no input augmentation. The ACMC loss on the
   AWGN-corrupted text embeddings is averaged over the strata, and the text generator,
   its latent code and the SNR encoder are updated.
3. **Auxiliary surrogate** (from epoch `--aux_warmup_epochs` on): the UCB bandit selects
   one auxiliary surrogate `s`. Both generators are conditioned on one SNR drawn
   uniformly from `[snr_min, snr_max]`, the ACMC loss is computed on the surrogate's
   adversarial embeddings without AWGN and on un-augmented images, and a separate step
   weighted by `lambda_sur * sigmoid(R_bar(s))` is taken. For a cross-system surrogate
   only the image optimizers step (image generator + SNR encoder, image latent code);
   for a same-system surrogate the image and text losses are summed and both the image
   and text optimizers step.

Outputs are written to `<save_dir>/<primary>/flickr30k/` (`/` in CLIP names becomes
`_`, e.g. `ViT-B_16`). At the end of every epoch the
generator checkpoints (`image-model-<epoch>.pth`, `text-model-<epoch>.pth`,
`image-z-<epoch>.pth`, `text-z-<epoch>.pth`), the UCB state (`aux_ucb-<epoch>.pth`) and
the perturbations produced by the primary image and text branches in the last iteration
(`uap_noise-<epoch>.pth`, `uap_embedding-<epoch>.pth`, one row per sample; resized to
224x224 for CLIP primaries) are saved. The evaluation deploys row 0 of `uap_noise` as
the image UAP and maps the mean of the `uap_embedding` rows to one adversarial word.
The feature-bank (`fb_*.pth`) and word-importance (`wi_*.pth`) caches are written to the
same directory on the first run and reused afterwards. Training can be resumed with
`--start_epoch <E> --load_dir <save_dir>`, which restores the generators, SNR encoder,
latent codes and UCB state of epoch `E-1` (not the Adam states or the random state).

## Evaluation

```bash
bash scripts/eval_ALBEF.sh
```

The UAPs of one source model are evaluated on the six target backbones of Table I at
SNR in {inf, 20, 15, 10, 5, 0} dB. I2T uses the adversarial image query against the
clean text gallery; T2I uses the adversarial text query against the adversarial image
gallery. The script prints the @1 metrics and saves to
`outputs/<output>/<source>_flickr30k_asr.xlsx` (one sheet per SNR plus a summary sheet)
the clean and adversarial Recall@1, the standard ASR@1/5/10 and the Pure UAP ASR@1/5/10
(Eq. (23)) for I2T and T2I. Pure UAP ASR takes its failure set at `gamma = inf` to
isolate the UAP's contribution from the channel's own effect on retrieval, and is
reported for the SNRs listed after `inf`, so keep `inf` first in `--snr_list`. The
`eval_*.sh` scripts use the default `--batch_size 4`, as in the paper runs;
finite-SNR results depend on the batch size through the random-stream order (and, in
`pixel` mode, the gallery noise power).

- `--target_models` restricts or reorders the targets and must include `--source_model`.
  The adversarial captions are generated when the evaluation reaches the source model,
  so targets listed before it are scored with clean captions, as in the paper runs.
- `--gallery_awgn {pixel,embedding}` selects where AWGN corrupts the adversarial T2I
  image gallery. `pixel` (default) is the setting used for the paper's tables:
  `apply_awgn` is applied to the UAP-perturbed gallery images before encoding, i.e.
  noise at the given SNR is added and each image row is then L2-normalized along the
  width. `embedding` applies it to the encoded gallery embeddings, following the channel
  model of Eqs. (9)-(10). At the first finite SNR the choice changes only the T2I
  adversarial Recall@1/ASR; the two modes consume the random streams differently, so
  every metric at later SNRs is a different noise realization. Pure UAP ASR takes its
  failure set at `gamma = inf`, so it does not depend on the gallery-noise model itself.

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{wang2026semcomuap,
  title     = {{SemCom-UAP}: Universal Adversarial Perturbations Against Cross-Modal Retrieval in Semantic Communications},
  author    = {Wang, Ziyu and Du, Yiming and Li, Jian and Ning, Rui and Hao, Shuai and Takabi, Daniel and Li, Lusi},
  booktitle = {2026 IEEE Conference on Communications and Network Security (CNS)},
  address   = {Newark, DE, USA},
  year      = {2026}
}
```

## Acknowledgements

The base pipeline, model wrappers and data loading are adapted from
[C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks):

```bibtex
@inproceedings{fang2025one,
  title     = {One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models},
  author    = {Fang, Hao and Kong, Jiawei and Yu, Wenbo and Chen, Bin and Li, Jiawei and Wu, Hao and Xia, Shu-Tao and Xu, Ke},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4090--4100},
  year      = {2025},
  doi       = {10.1109/ICCV51701.2025.00390}
}
```

We also thank the authors of [ALBEF](https://github.com/salesforce/ALBEF),
[TCL](https://github.com/uta-smile/TCL), [BLIP](https://github.com/salesforce/BLIP),
[X-VLM](https://github.com/zengyan-97/X-VLM) and [CLIP](https://github.com/openai/CLIP)
for releasing their code and models.

## License

The SemCom-UAP code is released under the [MIT License](LICENSE). The vendored
third-party code keeps its original license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
