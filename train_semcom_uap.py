"""
SemCom-UAP training entry point.

Trains the SNR-aware image/text UAP generators (generator.py) against a
primary surrogate VLP model, optimizing the adversarial cross-modal
contrastive loss under simulated AWGN channel corruption
(channel_aware_attacker.py), with stratified SNR sampling (--num_snr_strata,
paper symbol M) and UCB-based adaptive auxiliary-surrogate selection
(--aux_warmup_epochs, --aux_ucb_momentum, --aux_ucb_exploration).

Per iteration:
  1. image branch on the primary surrogate: M stratified SNR draws, backward,
     step of the image optimizers (G_x + f_snr, z_x);
  2. text branch on the primary surrogate: M fresh stratified SNR draws,
     backward, step of the text optimizers (G_t + f_snr, z_t);
  3. after the warm-up, one auxiliary surrogate s* is chosen by the UCB bandit
     and a separate weighted step is taken on its loss at one SNR drawn
     uniformly from [snr_min, snr_max] (image optimizers only for a
     cross-system surrogate, image and text optimizers for a same-system one).

At the end of every epoch the generator checkpoints and the perturbations
produced by the primary image and text branches in the last iteration of the
epoch (last SNR stratum, one row per latent row) are saved to
uap_noise-<epoch>.pth / uap_embedding-<epoch>.pth.

Usage:
python train_semcom_uap.py \
    --primary_encoder ALBEF \
    --primary_ckpt ./checkpoint/ALBEF/flickr30k.pth \
    --same_aux_encoders TCL XVLM BLIP \
    --same_aux_ckpts \
        ./checkpoint/TCL/flickr30k.pth \
        ./checkpoint/XVLM/flickr30k.pth \
        ./checkpoint/BLIP/flickr30k.pth \
    --cross_aux_encoders ViT-B/32 "ViT-B/16" \
    --cross_aux_ckpts "" "" \
    --save_dir output_semcom_uap \
    --batch_size 32 --epochs 100 --aux_warmup_epochs 40 \
    --snr_min 0 --snr_max 20 \
    --margin 0.02 --num_snr_strata 3
"""

import argparse
import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import ruamel.yaml as yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import utils
from utils import load_model
from dataset import paired_dataset
from generator import Generator, SNREncoder
from channel_aware_attacker import (
    ChannelAwareImageAttacker, ChannelAwareTextAttacker,
    acmc_loss, select_farthest_target,
    CLIP_ENCODERS, ENCODER_DIFFICULTY, SNR_EMB_DIM
)

BERT_ENCODERS = {'ALBEF', 'TCL', 'XVLM', 'BLIP'}

# ==========================================
# Arguments (paper symbols given in the help strings)
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument('--config',              default='./configs/Retrieval_flickr_train.yaml')
parser.add_argument('--seed',                default=42, type=int)
parser.add_argument('--batch_size',          default=64, type=int,
                    help='mini-batch size |B_x| = |B_t|')
parser.add_argument('--source_text_encoder', default='bert-base-uncased', type=str)
parser.add_argument('--save_dir',            type=str, default='output_semcom_uap')
parser.add_argument('--epochs',              type=int, default=100)
parser.add_argument('--lr',                  type=float, default=2e-4,
                    help='generator learning rate eta_g')
parser.add_argument('--start_epoch',         type=int, default=0)
parser.add_argument('--load_dir',            type=str, default=None)
parser.add_argument('--eps',                 type=int, default=12,
                    help='image budget epsilon_x, in units of 1/255')
parser.add_argument('--text_budget', '--word_num', dest='text_budget',
                    type=int, default=1,
                    help='text budget epsilon_t: number of perturbed token positions')

parser.add_argument('--primary_encoder',     type=str, required=True,
                    help='primary surrogate f^(1)')
parser.add_argument('--primary_ckpt',        type=str, default='')
parser.add_argument('--same_aux_encoders',   nargs='+', default=[],
                    help='same-system auxiliary surrogates (image + text supervision)')
parser.add_argument('--same_aux_ckpts',      nargs='+', default=[])
parser.add_argument('--cross_aux_encoders',  nargs='+', default=[],
                    help='cross-system auxiliary surrogates (image supervision only)')
parser.add_argument('--cross_aux_ckpts',     nargs='+', default=[])

parser.add_argument('--temperature',         type=float, default=0.07,
                    help='contrastive temperature tau (image branch)')
parser.add_argument('--txt_temperature',     type=float, default=0.07,
                    help='contrastive temperature tau (text branch)')
parser.add_argument('--aux_warmup_epochs',   type=int,   default=40,
                    help='warm-up length E_warm; no auxiliary step before this epoch')
parser.add_argument('--lambda_sur', '--lambda_aux_init', dest='lambda_sur',
                    type=float, default=0.65,
                    help='global auxiliary loss scale lambda_sur')
parser.add_argument('--aux_ucb_momentum',    type=float, default=0.1,
                    help='UCB reward EMA momentum eta')
parser.add_argument('--aux_ucb_exploration', type=float, default=1.0,
                    help='UCB exploration coefficient c_ucb')
parser.add_argument('--aux_min_select_ratio',type=float, default=0.02,
                    help='per-epoch selection floor ratio rho_min')
parser.add_argument('--wi_batch_size',       type=int,   default=256)
parser.add_argument('--z_lr_scale',          type=float, default=0.05,
                    help='latent-code learning-rate scale alpha (eta_z = alpha * eta_g)')

parser.add_argument('--snr_min',             type=float, default=0.0,
                    help='gamma_min in dB')
parser.add_argument('--snr_max',             type=float, default=20.0,
                    help='gamma_max in dB')
parser.add_argument('--margin',              type=float, default=0.02,
                    help='contrastive margin m')
parser.add_argument('--num_snr_strata', '--k_samples', dest='num_snr_strata',
                    type=int, default=3,
                    help='number of stratified SNR samples M per training step')

args = parser.parse_args()

assert len(args.same_aux_encoders)  == len(args.same_aux_ckpts)
assert len(args.cross_aux_encoders) == len(args.cross_aux_ckpts)

PRIMARY_IS_BERT = args.primary_encoder in BERT_ENCODERS
PRIMARY_IS_CLIP = args.primary_encoder in CLIP_ENCODERS
assert PRIMARY_IS_BERT or PRIMARY_IS_CLIP

if PRIMARY_IS_BERT:
    assert all(n in BERT_ENCODERS for n in args.same_aux_encoders)
    assert all(n in CLIP_ENCODERS for n in args.cross_aux_encoders)
else:
    assert all(n in CLIP_ENCODERS for n in args.same_aux_encoders)
    assert all(n in BERT_ENCODERS for n in args.cross_aux_encoders)

device = torch.device('cuda')
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

pname      = args.primary_encoder
pname_safe = pname.replace('/', '_')
save_dir   = os.path.join(args.save_dir, pname_safe, config['dataset'])
os.makedirs(save_dir, exist_ok=True)

FEAT_DIM    = 256 if PRIMARY_IS_BERT else 512
CTX_DIM     = FEAT_DIM + SNR_EMB_DIM   # 320 (BERT) or 576 (CLIP)
TXT_Z_SHAPE = (args.batch_size, 32, 1, 3) if PRIMARY_IS_BERT else (args.batch_size, 32, 1, 2)

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

normalization = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711))


# ==========================================
# Feature Bank
# ==========================================

class CrossModalFeatureBank:
    """Per-surrogate feature bank (Eq. (1)).

    Stores the normalized image feature v_i and the normalized feature u_i of
    the first caption T_{i,1} of every training image, in data-loader order.
    """

    def __init__(self, dataset_size, feat_dim):
        self.dataset_size = dataset_size
        self.feat_dim     = feat_dim
        self.img_features = torch.zeros(dataset_size, feat_dim)
        self.txt_features = torch.zeros(dataset_size, feat_dim)
        self.count        = 0

    @torch.no_grad()
    def build_bert(self, model, tokenizer, data_loader, device):
        model.eval()
        idx = 0
        print("  [FeatureBank:BERT] Extracting...")
        for _, (images, texts_group, _, _) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            bs     = images.shape[0]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                img_f = F.normalize(
                    model.inference_image(normalization(images))['image_feat'].float(), dim=-1)
                txt_f = F.normalize(
                    model.inference_text(
                        tokenizer([texts_group[i][0] for i in range(bs)],
                                  padding='max_length', truncation=True,
                                  max_length=30, return_tensors='pt').to(device)
                    )['text_feat'].float(), dim=-1)
            self.img_features[idx:idx+bs] = img_f.cpu()
            self.txt_features[idx:idx+bs] = txt_f.cpu()
            idx += bs
        self.count = idx
        print(f"  [FeatureBank:BERT] Done. {idx} samples.")

    @torch.no_grad()
    def build_clip(self, model, data_loader, device):
        from models.clip_model import clip as _clip
        model.eval()
        clip_size = model.visual.input_resolution
        idx = 0
        print("  [FeatureBank:CLIP] Extracting...")
        for _, (images, texts_group, _, _) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            bs     = images.shape[0]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                imgs_in = F.interpolate(images, size=(clip_size, clip_size),
                                        mode='bilinear', align_corners=False) \
                          if images.shape[-1] != clip_size else images
                img_f = F.normalize(model.encode_image(normalization(imgs_in)).float(), dim=-1)
                txt_f = F.normalize(
                    model.encode_text(
                        _clip.tokenize([texts_group[i][0] for i in range(bs)],
                                       truncate=True).to(device)).float(), dim=-1)
            self.img_features[idx:idx+bs] = img_f.cpu()
            self.txt_features[idx:idx+bs] = txt_f.cpu()
            idx += bs
        self.count = idx
        print(f"  [FeatureBank:CLIP] Done. {idx} samples.")

    def save(self, path):
        torch.save({'img_features': self.img_features[:self.count],
                    'txt_features': self.txt_features[:self.count],
                    'count': self.count, 'feat_dim': self.feat_dim}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu')
        self.count, self.feat_dim = d['count'], d['feat_dim']
        self.img_features = torch.zeros(self.dataset_size, self.feat_dim)
        self.txt_features = torch.zeros(self.dataset_size, self.feat_dim)
        self.img_features[:self.count] = d['img_features']
        self.txt_features[:self.count] = d['txt_features']
        print(f"  [FeatureBank] Loaded ({self.count} samples, dim={self.feat_dim})")


# ==========================================
# Word Importance Cache
# ==========================================

class WordImportanceCacheBERT:
    def __init__(self):
        self.all_positions     = {}
        self.all_padding_zeros = {}
        self.total_texts       = 0

    @torch.no_grad()
    def build(self, model, tokenizer, data_loader, num_perturbation,
              max_length, device, bsi=256):
        model.eval()
        g = 0
        for _, (_, texts_group, _, _) in enumerate(tqdm(data_loader, desc="  WI:BERT")):
            texts = [t for tg in texts_group for t in tg]
            if not texts: continue
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ti = tokenizer(texts, padding='max_length', truncation=True,
                               max_length=max_length, return_tensors='pt').to(device)
                oe = model.inference_text(ti)['text_feat'].flatten(1).float().detach()
            all_m, offsets, wcounts = [], [], []
            for text in texts:
                words = text.split(' ')
                s = len(all_m)
                for i in range(len(words)):
                    all_m.append(' '.join(words[:i] + ['[UNK]'] + words[i+1:]))
                offsets.append((s, len(all_m)))
                wcounts.append(len(words))
            all_me = []
            for i in range(0, len(all_m), bsi):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    mi = tokenizer(all_m[i:i+bsi], padding='max_length', truncation=True,
                                   max_length=max_length, return_tensors='pt').to(device)
                    me = model.inference_text(mi)['text_feat'].flatten(1).float().detach()
                all_me.append(me)
            all_me = torch.cat(all_me, dim=0)
            crit = torch.nn.KLDivLoss(reduction='none')
            for li, text in enumerate(texts):
                gi   = g + li
                s, e = offsets[li]
                nw   = wcounts[li]
                sc   = crit(all_me[s:e].log_softmax(dim=-1),
                            oe[li].softmax(dim=-1).unsqueeze(0).expand(nw, -1)).sum(dim=-1)
                loi  = sorted(enumerate(sc.cpu().tolist()), key=lambda x: x[1], reverse=True)
                words = text.split(' ')
                keys, idx = [], 0
                for w in words:
                    sub = tokenizer.tokenize(w)
                    keys.append([idx, idx + len(sub)])
                    idx += len(sub)
                pos, pz = [], []
                for j in range(num_perturbation):
                    if j < len(loi) and loi[j][0] < len(keys):
                        pos.append(keys[loi[j][0]][0])
                        pz.append(keys[loi[j][0]][1] - keys[loi[j][0]][0] - 1)
                self.all_positions[gi]     = pos
                self.all_padding_zeros[gi] = pz
            g += len(texts)
        self.total_texts = g

    def get_batch(self, indices):
        return ([self.all_positions.get(i, []) for i in indices],
                [self.all_padding_zeros.get(i, []) for i in indices])

    def save(self, path):
        torch.save({'all_positions': self.all_positions,
                    'all_padding_zeros': self.all_padding_zeros,
                    'total_texts': self.total_texts}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu')
        self.all_positions     = d['all_positions']
        self.all_padding_zeros = d['all_padding_zeros']
        self.total_texts       = d['total_texts']
        print(f"  [WI:BERT] Loaded ({self.total_texts} texts)")


class WordImportanceCacheCLIP:
    def __init__(self):
        self.all_positions     = {}
        self.all_padding_zeros = {}
        self.total_texts       = 0

    @torch.no_grad()
    def build(self, model, data_loader, num_perturbation, device, bsi=256):
        from models.clip_model import clip as _clip
        model.eval()
        g = 0
        for _, (_, texts_group, _, _) in enumerate(tqdm(data_loader, desc="  WI:CLIP")):
            texts = [t for tg in texts_group for t in tg]
            if not texts: continue
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                oe = model.encode_text(
                    _clip.tokenize(texts, truncate=True).to(device)).float().detach()
            all_m, offsets, wcounts = [], [], []
            for text in texts:
                words = text.split(' ')
                s = len(all_m)
                for i in range(len(words)):
                    all_m.append(' '.join(words[:i] + ['[UNK]'] + words[i+1:]))
                offsets.append((s, len(all_m)))
                wcounts.append(len(words))
            all_me = []
            for i in range(0, len(all_m), bsi):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    me = model.encode_text(
                        _clip.tokenize(all_m[i:i+bsi], truncate=True).to(device)).float().detach()
                all_me.append(me)
            all_me = torch.cat(all_me, dim=0)
            crit = torch.nn.KLDivLoss(reduction='none')
            for li, text in enumerate(texts):
                gi   = g + li
                s, e = offsets[li]
                nw   = wcounts[li]
                sc   = crit(all_me[s:e].log_softmax(dim=-1),
                            oe[li].softmax(dim=-1).unsqueeze(0).expand(nw, -1)).sum(dim=-1)
                loi  = sorted(enumerate(sc.cpu().tolist()), key=lambda x: x[1], reverse=True)
                pos  = []
                for j in range(num_perturbation):
                    if j < len(loi):
                        pos.append(loi[j][0] + 1)
                self.all_positions[gi]     = pos
                self.all_padding_zeros[gi] = []
            g += len(texts)
        self.total_texts = g

    def get_batch(self, indices):
        return ([self.all_positions.get(i, []) for i in indices],
                [self.all_padding_zeros.get(i, []) for i in indices])

    def save(self, path):
        torch.save({'all_positions': self.all_positions,
                    'all_padding_zeros': self.all_padding_zeros,
                    'total_texts': self.total_texts}, path)

    def load(self, path):
        d = torch.load(path, map_location='cpu')
        self.all_positions     = d['all_positions']
        self.all_padding_zeros = d['all_padding_zeros']
        self.total_texts       = d['total_texts']
        print(f"  [WI:CLIP] Loaded ({self.total_texts} texts)")


# ==========================================
# AuxiliaryUCB
# ==========================================

class AuxiliaryUCB:
    """UCB-based adaptive surrogate selection (Section IV-G, Eqs. (16)-(19)).

    Rewards are z-score normalized per surrogate so that losses from
    heterogeneous model families are comparable. Reward estimates and
    selection counts start at zero; the running mean of the raw loss of
    surrogate s starts at 3 * ENCODER_DIFFICULTY[s] (difficulty 0.40, i.e. a
    starting mean of 1.2, for unlisted names).
    """

    def __init__(self, names, lambda_sur=0.65,
                 ucb_momentum=0.1, exploration_coeff=1.0,
                 running_momentum=0.99, min_select_ratio=0.02):
        self.names             = names
        self.n                 = len(names)
        self.lambda_sur        = lambda_sur
        self.ucb_momentum      = ucb_momentum
        self.exploration_coeff = exploration_coeff
        self.running_momentum  = running_momentum
        self.min_select_ratio  = min_select_ratio
        self.R            = np.zeros(self.n, dtype=np.float64)   # R_bar(s) <- 0
        self.T            = np.zeros(self.n, dtype=np.float64)   # n_s      <- 0
        self.total        = 0
        init_means        = np.array([ENCODER_DIFFICULTY.get(n, 0.40) * 3.0
                                       for n in names], dtype=np.float64)
        self.running_mean = init_means.copy()                    # mu_s
        self.running_var  = np.ones(self.n, dtype=np.float64)    # sigma_s^2
        self._epoch_counts   = np.zeros(self.n, dtype=np.int64)
        self._min_select_abs = 0

    def set_epoch_batches(self, n_batches):
        self._epoch_counts   = np.zeros(self.n, dtype=np.int64)
        self._min_select_abs = max(1, int(n_batches * self.min_select_ratio))

    def _normalize(self, idx, raw_loss):
        m, v = self.running_mean[idx], self.running_var[idx]
        mom  = self.running_momentum
        self.running_mean[idx] = mom * m + (1 - mom) * raw_loss
        self.running_var[idx]  = mom * v + (1 - mom) * (raw_loss - m) ** 2
        return float((raw_loss - self.running_mean[idx]) / (np.sqrt(self.running_var[idx]) + 1e-6))

    def _ucb_scores(self, indices):
        return (self.R[indices] + self.exploration_coeff *
                np.sqrt(np.log(self.total + 1) / (self.T[indices] + 1e-8)))

    def select(self):
        if self.n == 0: return None
        if self.n == 1: self._epoch_counts[0] += 1; return 0
        cands = np.where(self._epoch_counts < self._min_select_abs)[0] \
                if self._min_select_abs > 0 else np.array([], dtype=np.int64)
        idx = int(cands[np.argmax(self._ucb_scores(cands))]) if len(cands) > 0 \
              else int(np.argmax(self._ucb_scores(np.arange(self.n))))
        self._epoch_counts[idx] += 1
        return idx

    def get_weight(self, idx):
        """Sigmoid-gated auxiliary loss weight, lambda_sur * sigmoid(R_bar(s))."""
        if self.n == 0: return 0.0
        return float(self.lambda_sur / (1.0 + np.exp(-float(self.R[idx]))))

    def update(self, idx, raw_loss):
        self.R[idx] = (1 - self.ucb_momentum) * self.R[idx] + \
                       self.ucb_momentum * self._normalize(idx, raw_loss)
        self.T[idx] += 1; self.total += 1

    def summary(self):
        lines = ["AuxiliaryUCB state:"]
        for i, n in enumerate(self.names):
            lines.append(f"  {n:12s}  R={self.R[i]:+.4f}  T={int(self.T[i]):5d}"
                         f"  epoch_cnt={self._epoch_counts[i]:4d}"
                         f"  weight={self.get_weight(i):.4f}")
        return '\n'.join(lines)

    def state_dict(self):
        return {'R': self.R.copy(), 'T': self.T.copy(), 'total': self.total,
                'names': self.names, 'running_mean': self.running_mean.copy(),
                'running_var': self.running_var.copy()}

    def load_state_dict(self, d):
        self.R            = d['R'].copy(); self.T = d['T'].copy()
        self.total        = d['total']
        self.running_mean = d.get('running_mean', self.running_mean).copy()
        self.running_var  = d.get('running_var',  self.running_var).copy()


# ==========================================
# Auxiliary-surrogate losses (Eq. (15))
# ==========================================
# The auxiliary losses are computed on the clean (not channel-corrupted)
# adversarial embeddings of the selected surrogate; the SNR only enters through
# the generator condition.

def compute_aux_img_loss(model, adv_imgs_norm, clean_img, clean_txt,
                         is_clip, temperature, margin, device):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        adv_f = F.normalize(model.encode_image(adv_imgs_norm).float(), dim=-1) \
                if is_clip else \
                F.normalize(model.inference_image(adv_imgs_norm)['image_feat'].float(), dim=-1)
        clean_v = F.normalize(clean_img.to(device), dim=-1)
        clean_u = F.normalize(clean_txt.to(device), dim=-1)
        t_img, t_txt = select_farthest_target(adv_f, clean_v, clean_u)
        loss = acmc_loss(adv_f, t_img, t_txt, clean_v, clean_u,
                         temperature=temperature, margin=margin)
        return loss.float()


def compute_aux_txt_loss(model, tokenizer, texts, txt2img, uap_emb,
                         clean_img, clean_txt, cached_positions, cached_padding_zeros,
                         is_clip, temperature, margin, device):
    """uap_emb: (B, D) row-wise norm-clamped text perturbation; a CLIP
    surrogate receives row 0 only, a BERT-family surrogate all rows."""
    ti = tokenizer(texts, padding='max_length', truncation=True,
                   max_length=30, return_tensors='pt').to(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _uap_e  = uap_emb[0:1] if is_clip else uap_emb
        ao      = model.inference_text_replace(ti, _uap_e, cached_positions, cached_padding_zeros)
        adv_f   = F.normalize(ao['text_feat'].flatten(1).float(), dim=-1)
        clean_v = F.normalize(clean_img[txt2img].to(device), dim=-1)
        clean_u = F.normalize(clean_txt[txt2img].to(device), dim=-1)
        t_img, t_txt = select_farthest_target(adv_f, clean_v, clean_u)
        loss = acmc_loss(adv_f, t_img, t_txt, clean_v, clean_u,
                         temperature=temperature, margin=margin)
        return loss.float()


# ==========================================
# Main training routine
# ==========================================

def train(primary_cfg, same_aux_cfgs, cross_aux_cfgs, data_loader, device, args):
    dataset_size = len(data_loader.dataset)
    n_batches    = len(data_loader)

    # Images / captions one pass over the (unshuffled) loader yields; cached
    # feature banks and word-importance tables built with a different batch
    # layout are rebuilt instead of being reused with missing entries.
    n_loader_imgs = n_batches * data_loader.batch_size if data_loader.drop_last \
                    else dataset_size
    n_loader_txts = sum(len(data_loader.dataset.img2txt[i]) for i in range(n_loader_imgs))

    print("=" * 60)
    print(f"SemCom-UAP Train: Primary={pname} ({'BERT' if PRIMARY_IS_BERT else 'CLIP'})")
    print(f"  Same-system aux: {[c['name'] for c in same_aux_cfgs]}")
    print(f"  Cross-system aux: {[c['name'] for c in cross_aux_cfgs]}")
    print(f"  FEAT_DIM={FEAT_DIM}  CTX_DIM={CTX_DIM}  SNR_EMB={SNR_EMB_DIM}")
    print(f"  M={args.num_snr_strata}  SNR=[{args.snr_min},{args.snr_max}]dB  margin={args.margin}")
    print(f"  eps={args.eps}/255  epochs={args.epochs}  warmup={args.aux_warmup_epochs}")
    print("=" * 60)

    for cfg in [primary_cfg] + same_aux_cfgs + cross_aux_cfgs:
        cfg['model'].float().eval()
        for p in cfg['model'].parameters():
            p.requires_grad_(False)

    # ── Generator + SNREncoder + z ────────────────────────────────────
    img_G      = Generator(100, [[1024, 512], [256, 128], [64, 32]], 3,
                           args.batch_size, first_kernel_size=4,
                           num_heads=1, context_dim=CTX_DIM)
    txt_G      = Generator(32, [[256, 128], [64, 32]], args.text_budget,
                           args.batch_size, first_kernel_size=1,
                           num_heads=1, context_dim=CTX_DIM)

    # the SNR encoder f_snr is shared by both generators
    snr_enc = SNREncoder(snr_emb_dim=SNR_EMB_DIM)

    img_z_init = torch.randn(args.batch_size, 100, 3, 3)
    txt_z_init = torch.randn(*TXT_Z_SHAPE)

    if PRIMARY_IS_BERT:
        _we_file = f"word_embedding_{pname}.pth"
        if not os.path.exists(_we_file):
            _we_file = "word_embedding_ALBEF.pth"
    else:
        _we_file = "word_embedding_CLIP.pth"
    we    = torch.load(_we_file).to(device).detach()
    norms = torch.norm(we, p=2, dim=1)

    img_G.to(device); txt_G.to(device); snr_enc.to(device)

    if args.start_epoch > 0 and args.load_dir:
        ld = os.path.join(args.load_dir, pname_safe, config['dataset'])
        ep = args.start_epoch - 1
        img_z_init = torch.load(os.path.join(ld, f'image-z-{ep}.pth'), map_location='cpu')
        txt_z_init = torch.load(os.path.join(ld, f'text-z-{ep}.pth'), map_location='cpu')

    img_z = img_z_init.to(device)
    txt_z = txt_z_init.to(device)

    # Each attacker owns an Adam over its generator + the shared f_snr (lr
    # eta_g) and an Adam over its latent code (lr eta_z = alpha * eta_g),
    # both with betas (0.5, 0.999).
    img_atk = ChannelAwareImageAttacker(
        generator=img_G, snr_encoder=snr_enc, z_init=img_z,
        eps=args.eps/255, device=device, lr=args.lr,
        temperature=args.temperature, z_lr_scale=args.z_lr_scale,
        snr_min_db=args.snr_min, snr_max_db=args.snr_max,
        margin=args.margin, num_snr_strata=args.num_snr_strata)

    txt_atk = ChannelAwareTextAttacker(
        ref_model_name=pname,
        generator=txt_G, snr_encoder=snr_enc, z_init=txt_z,
        device=device, temperature=args.txt_temperature,
        min_norm=norms.min(), max_norm=norms.max(),
        lr=args.lr, max_length=30,
        text_budget=args.text_budget, z_lr_scale=args.z_lr_scale,
        snr_min_db=args.snr_min, snr_max_db=args.snr_max,
        margin=args.margin, num_snr_strata=args.num_snr_strata)

    if args.start_epoch > 0 and args.load_dir:
        ld = os.path.join(args.load_dir, pname_safe, config['dataset'])
        ep = args.start_epoch - 1
        img_atk.load_model(os.path.join(ld, f'image-model-{ep}.pth'), device)
        txt_atk.load_model(os.path.join(ld, f'text-model-{ep}.pth'),  device)

    # ── Feature Bank ──────────────────────────────────────────────────
    def _build_fb(cfg, tag):
        name     = cfg['name']
        safe     = name.replace('/', '_')
        is_clip  = name in CLIP_ENCODERS
        feat_dim = 512 if is_clip else 256
        fb_path  = os.path.join(save_dir, f'fb_{tag}_{safe}.pth')
        fb       = CrossModalFeatureBank(dataset_size, feat_dim=feat_dim)
        cached   = False
        if os.path.exists(fb_path):
            print(f"\n  Loading cached FB [{name}]: {fb_path}")
            fb.load(fb_path)
            cached = (fb.count == n_loader_imgs)
            if not cached:
                print(f"  [FeatureBank] Cache has {fb.count} samples, the loader "
                      f"yields {n_loader_imgs}; rebuilding.")
                fb = CrossModalFeatureBank(dataset_size, feat_dim=feat_dim)
        if not cached:
            if is_clip: fb.build_clip(cfg['model'], data_loader, device)
            else:       fb.build_bert(cfg['model'], cfg['tokenizer'], data_loader, device)
            fb.save(fb_path)
        torch.cuda.empty_cache()
        return fb

    print(f"\n=== Building Primary FB: {pname} ===")
    primary_fb = _build_fb(primary_cfg, 'primary')
    same_fbs   = [_build_fb(cfg, 'sameaux')  for cfg in same_aux_cfgs]
    cross_fbs  = [_build_fb(cfg, 'crossaux') for cfg in cross_aux_cfgs]

    # ── Word Importance ───────────────────────────────────────────────
    def _build_wi(cfg, tag):
        name    = cfg['name']
        safe    = name.replace('/', '_')
        is_clip = name in CLIP_ENCODERS
        wi_path = os.path.join(save_dir, f'wi_{tag}_{safe}.pth')
        wi      = WordImportanceCacheCLIP() if is_clip else WordImportanceCacheBERT()
        cached  = False
        if os.path.exists(wi_path):
            wi.load(wi_path)
            cached = (wi.total_texts == n_loader_txts)
            if not cached:
                print(f"  [WI] Cache has {wi.total_texts} texts, the loader "
                      f"yields {n_loader_txts}; rebuilding.")
                wi = WordImportanceCacheCLIP() if is_clip else WordImportanceCacheBERT()
        if not cached:
            if is_clip: wi.build(cfg['model'], data_loader, args.text_budget, device, args.wi_batch_size)
            else:       wi.build(cfg['model'], cfg['tokenizer'], data_loader,
                                  args.text_budget, 30, device, args.wi_batch_size)
            wi.save(wi_path)
        return wi

    primary_wi = _build_wi(primary_cfg, 'primary')
    same_wis   = [_build_wi(cfg, 'sameaux') for cfg in same_aux_cfgs]

    # ── Auxiliary UCB ─────────────────────────────────────────────────
    all_aux_names = [c['name'] for c in same_aux_cfgs] + [c['name'] for c in cross_aux_cfgs]
    n_same_aux    = len(same_aux_cfgs)

    aux_ucb = AuxiliaryUCB(
        names=all_aux_names, lambda_sur=args.lambda_sur,
        ucb_momentum=args.aux_ucb_momentum, exploration_coeff=args.aux_ucb_exploration,
        min_select_ratio=args.aux_min_select_ratio)

    if args.start_epoch > 0 and args.load_dir:
        ucb_path = os.path.join(args.load_dir, pname_safe, config['dataset'],
                                f'aux_ucb-{args.start_epoch-1}.pth')
        if os.path.exists(ucb_path):
            aux_ucb.load_state_dict(torch.load(ucb_path, map_location='cpu'))

    # ── Training Loop ─────────────────────────────────────────────────
    for epoch in range(args.start_epoch, args.epochs):
        p_img_loss = p_txt_loss = a_loss_sum = 0.0
        lc = 0
        aux_usage = [0] * len(all_aux_names)
        g_si = g_ti = 0
        last_ue = None

        if len(all_aux_names) > 0 and epoch >= args.aux_warmup_epochs:
            aux_ucb.set_epoch_batches(n_batches)

        for bi, (images, texts_group, _, _) in enumerate(tqdm(data_loader)):
            texts, img2txt, txt2img = [], [], []
            tid = 0
            for i in range(len(texts_group)):
                texts   += texts_group[i]
                txt2img += [i] * len(texts_group[i])
                img2txt.append([])
                for j in range(len(texts_group[i])):
                    img2txt[i].append(tid); tid += 1

            images = images.to(device)
            bs     = images.shape[0]
            b_idx  = list(range(g_si, g_si + bs));         g_si += bs
            bt_idx = list(range(g_ti, g_ti + len(texts))); g_ti += len(texts)

            # The image generator is conditioned on the first-caption feature
            # of each image (one copy per caption, averaged per image inside
            # the attacker); the text generator on the per-image feature.
            ref_txt_feats = F.normalize(primary_fb.txt_features[b_idx].to(device), dim=-1)
            ref_img_feats = F.normalize(primary_fb.img_features[b_idx].to(device), dim=-1)
            src_img       = primary_fb.img_features[b_idx]
            src_txt       = primary_fb.txt_features[b_idx]

            ref_txt_embeds_full = torch.stack(
                [ref_txt_feats[i] for i in range(len(img2txt))
                 for _ in img2txt[i]], dim=0)

            cp, cpz = (primary_wi.get_batch(bt_idx)
                       if primary_wi.total_texts > 0
                       else ([[] for _ in bt_idx], [[] for _ in bt_idx]))

            # ── Primary image branch (own step) ──────────────────────
            p_il, uap_noise = img_atk.attack(
                model=primary_cfg['model'], enc_name=pname,
                normalization=normalization,
                images=images, img2txt=img2txt,
                ref_txt_feats=ref_txt_embeds_full,
                clean_img_feats=src_img, clean_txt_feats=src_txt)

            # ── Primary text branch (own step) ───────────────────────
            p_tl, uap_embedding = txt_atk.attack(
                model=primary_cfg['model'],
                tokenizer=primary_cfg['tokenizer'],
                texts=texts, txt2img=txt2img,
                ref_img_feats=ref_img_feats,
                clean_img_feats=src_img, clean_txt_feats=src_txt,
                cached_positions=cp, cached_padding_zeros=cpz)

            # ── Auxiliary surrogate s* (after warm-up, separate step) ─
            if len(all_aux_names) > 0 and epoch >= args.aux_warmup_epochs:
                ai         = aux_ucb.select()
                aux_weight = aux_ucb.get_weight(ai)
                aux_usage[ai] += 1
                is_cross   = (ai >= n_same_aux)

                img_atk.optimG.zero_grad(); img_atk.optimZ.zero_grad()
                txt_atk.optimG.zero_grad(); txt_atk.optimZ.zero_grad()

                # one uniformly drawn SNR conditions the auxiliary UAPs
                aux_snr = np.random.uniform(args.snr_min, args.snr_max)

                if is_cross:
                    # cross-system: image branch only (Eq. (15))
                    ci          = ai - n_same_aux
                    aux_cfg     = cross_aux_cfgs[ci]
                    aux_fb      = cross_fbs[ci]
                    aux_is_clip = aux_cfg['name'] in CLIP_ENCODERS

                    uap_a    = img_atk.generate_uap(ref_txt_feats, aux_snr)
                    adv_norm = normalization(img_atk.apply_uap(
                        images, uap_a, aux_cfg['model'], aux_is_clip))
                    aux_img_loss = compute_aux_img_loss(
                        aux_cfg['model'], adv_norm,
                        aux_fb.img_features[b_idx], aux_fb.txt_features[b_idx],
                        aux_is_clip, args.temperature, args.margin, device)

                    # Eq. (19): sigmoid-gated weighting of the selected surrogate
                    aux_loss = aux_weight * aux_img_loss
                    aux_loss.backward()
                    img_atk.optimG.step(); img_atk.optimZ.step()
                    aux_ucb.update(ai, aux_img_loss.item())

                else:
                    # same-system: image + text branches (Eq. (15))
                    si          = ai
                    aux_cfg     = same_aux_cfgs[si]
                    aux_fb      = same_fbs[si]
                    aux_wi      = same_wis[si]
                    aux_is_clip = aux_cfg['name'] in CLIP_ENCODERS

                    aux_cp, aux_cpz = (aux_wi.get_batch(bt_idx)
                                       if aux_wi.total_texts > 0
                                       else ([[] for _ in bt_idx], [[] for _ in bt_idx]))

                    uap_a    = img_atk.generate_uap(ref_txt_feats, aux_snr)
                    adv_norm = normalization(img_atk.apply_uap(
                        images, uap_a, aux_cfg['model'], aux_is_clip))
                    aux_img_loss = compute_aux_img_loss(
                        aux_cfg['model'], adv_norm,
                        aux_fb.img_features[b_idx], aux_fb.txt_features[b_idx],
                        aux_is_clip, args.temperature, args.margin, device)

                    ue_a = txt_atk.generate_uap(ref_img_feats, aux_snr)
                    aux_txt_loss = compute_aux_txt_loss(
                        aux_cfg['model'], aux_cfg['tokenizer'],
                        texts, txt2img, ue_a,
                        aux_fb.img_features[b_idx], aux_fb.txt_features[b_idx],
                        aux_cp, aux_cpz, aux_is_clip,
                        args.txt_temperature, args.margin, device)

                    # Eq. (19): sigmoid-gated weighting of the selected surrogate
                    aux_loss = aux_weight * (aux_img_loss + aux_txt_loss)
                    aux_loss.backward()
                    img_atk.optimG.step(); img_atk.optimZ.step()
                    txt_atk.optimG.step(); txt_atk.optimZ.step()
                    aux_ucb.update(ai, (aux_img_loss.item() + aux_txt_loss.item()) / 2.0)

                a_loss_sum += aux_loss.item()

            last_ue = uap_embedding.detach() if uap_embedding is not None else last_ue
            p_img_loss += p_il.item()
            p_txt_loss += p_tl.item()
            lc += 1

            if (bi + 1) % 10 == 0:
                au = ' '.join([f"{all_aux_names[i]}:{aux_usage[i]}"
                               for i in range(len(all_aux_names))])
                wt = (f'[WARMUP {epoch+1}/{args.aux_warmup_epochs}]'
                      if epoch < args.aux_warmup_epochs else '[AUX ON ]')
                print(f'E{epoch} {wt} B{bi+1}/{n_batches} | '
                      f'P_IMG:{p_img_loss/lc:.4f} P_TXT:{p_txt_loss/lc:.4f} | '
                      f'AUX:{a_loss_sum/lc:.4f} | [{au}]')
                p_img_loss = p_txt_loss = a_loss_sum = 0.0; lc = 0

        # ── Checkpoint + perturbation export ─────────────────────────
        img_atk.save_model(f'{save_dir}/image-model-{epoch}.pth')
        img_atk.save_z(    f'{save_dir}/image-z-{epoch}.pth')
        txt_atk.save_model(f'{save_dir}/text-model-{epoch}.pth')
        txt_atk.save_z(    f'{save_dir}/text-z-{epoch}.pth')
        # Perturbations of the primary branches in the last iteration (last SNR
        # stratum), one row per latent row: uap_noise (B, 3, H, W), H = W = the CLIP input
        # resolution for a CLIP primary; uap_embedding (B, D).
        torch.save(uap_noise.float() if uap_noise is not None else None, f'{save_dir}/uap_noise-{epoch}.pth')
        if last_ue is not None:
            torch.save(last_ue.float(), f'{save_dir}/uap_embedding-{epoch}.pth')
        torch.save(aux_ucb.state_dict(), f'{save_dir}/aux_ucb-{epoch}.pth')
        print(f"Epoch {epoch} saved.")
        print(aux_ucb.summary())

    torch.cuda.empty_cache()
    print("\nTraining complete!")


# ==========================================
# Entry point
# ==========================================

if __name__ == '__main__':
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print(f"\nLoading primary encoder: {pname}")
    p_model, _, p_tokenizer = load_model(
        pname, args.primary_ckpt, args.source_text_encoder, config, device)
    p_model.to(device)
    primary_cfg = {'name': pname, 'model': p_model, 'tokenizer': p_tokenizer}

    same_aux_cfgs = []
    for name, ckpt in zip(args.same_aux_encoders, args.same_aux_ckpts):
        model, _, tok = load_model(name, ckpt, args.source_text_encoder, config, device)
        model.to(device)
        same_aux_cfgs.append({'name': name, 'model': model, 'tokenizer': tok})

    cross_aux_cfgs = []
    for name, ckpt in zip(args.cross_aux_encoders, args.cross_aux_ckpts):
        model, _, tok = load_model(name, ckpt, args.source_text_encoder, config, device)
        model.to(device)
        cross_aux_cfgs.append({'name': name, 'model': model, 'tokenizer': tok})

    print("\nCreating dataset")
    s_tf = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor()])
    ds = paired_dataset(config['annotation_file'], s_tf, config['image_root'])
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4,
                    collate_fn=ds.collate_fn, drop_last=True, shuffle=False)

    train(primary_cfg, same_aux_cfgs, cross_aux_cfgs, dl, device, args)
