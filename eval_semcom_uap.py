"""
SemCom-UAP evaluation: cross-modal retrieval under a simulated AWGN channel.

I2T: adversarial image (UAP + AWGN) queried against a clean text gallery.
T2I: adversarial text (token substitution + AWGN) queried against an
     adversarial image gallery (also UAP + AWGN corrupted).

The text-side token substitution reuses the nearest-neighbor word-embedding
lookup + TextAttacker.get_adv_text substitution from the base C-PGC pipeline
(word_idx_{model}.npy for BERT-based encoders, word_idx_CLIP.npy for CLIP).

Clean and adversarial queries are encoded first; the transmitted embeddings
are then corrupted by AWGN. For the adversarial image gallery, --gallery_awgn
selects where the channel noise enters:
    pixel      (default; the setting of the paper's tables) apply_awgn on the
               UAP-perturbed gallery images before encoding. apply_awgn
               re-normalizes along the last dimension, so each image row is
               rescaled to unit L2 norm.
    embedding  apply_awgn on the encoded gallery embeddings, i.e. the channel
               model of Eqs. (9)-(10).
At the first finite SNR in --snr_list the choice changes only the T2I
adversarial Recall@1 / ASR. The two modes draw the gallery noise at different
points of the random stream (on GPU, pixel mode draws it from the CUDA
generator, embedding mode from the CPU generator that also serves the query
AWGN), so every metric at the later SNRs, clean Recall@1 and Pure UAP ASR
included, is a different noise realization. Pure UAP ASR does not depend on
the gallery-noise model itself, since its failure set is taken at gamma = inf,
where the channel is noiseless.

Reports Recall@1, standard ASR, and Pure UAP ASR (Eq. (23)), whose failure set
is fixed at gamma = inf so that it measures the UAP alone:
    ASR(gamma)          = |clean_success(gamma) & adv_fail(gamma)| / |clean_success(gamma)|
    Pure UAP ASR(gamma) = |clean_success(gamma) & adv_fail(inf)|   / |clean_success(gamma)|
Pure UAP ASR is reported for the SNRs evaluated after 'inf' in --snr_list.
"""

import argparse
import gc
import os
from ruamel.yaml import YAML
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.clip_model.simple_tokenizer import SimpleTokenizer
import utils
from utils import load_model, get_filter_words
from attacker import TextAttacker
from dataset import paired_dataset
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config',   default='./configs/Retrieval_flickr_test.yaml')
parser.add_argument('--seed',     default=42, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--text_encoder', default='bert-base-uncased', type=str)
parser.add_argument('--source_model', default='ALBEF', type=str)
parser.add_argument('--checkpoint',   default='./checkpoint', type=str)
parser.add_argument('--load_dir', type=str)
parser.add_argument('--epoch',    type=int, default=99)
parser.add_argument('--text_budget', '--word_num', dest='text_budget',
                    type=int, default=1,
                    help='text budget epsilon_t, passed to TextAttacker as '
                         'number_perturbation; get_adv_text substitutes one '
                         'token per adversarial word, and the exported '
                         'uap_embedding is averaged into a single word')
parser.add_argument('--output',   type=str, default='result_semcom_asr')
parser.add_argument('--snr_list', nargs='+', default=['inf', '20', '15', '10', '5', '0'],
                    help='channel SNRs in dB; "inf" denotes a noiseless channel')
parser.add_argument('--target_models', nargs='+',
                    default=['ALBEF', 'TCL', 'BLIP', 'XVLM', 'ViT-B/16', 'ViT-B/32'],
                    help='evaluation VLP models (columns of Table I), '
                         'evaluated in the given order; must include --source_model')
parser.add_argument('--gallery_awgn', type=str, default='pixel',
                    choices=['pixel', 'embedding'],
                    help='where AWGN corrupts the adversarial T2I image gallery: '
                         '"pixel" (paper tables) on the UAP-perturbed images '
                         'before encoding, "embedding" on the encoded gallery '
                         'embeddings (Eqs. (9)-(10))')
args = parser.parse_args()
if args.source_model not in args.target_models:
    parser.error(f'--target_models must include --source_model '
                 f'({args.source_model}): the adversarial captions are '
                 f'generated when the evaluation reaches the source model')

filter_words = get_filter_words()

_EVAL_TAG_MAP = {
    'ALBEF':    'ALBEF',
    'TCL':      'TCL',
    'BLIP':     'BLIP',
    'XVLM':     'XVLM',
    'X-VLM':    'XVLM',
    'ViT-B/16': 'ViT-B_16',
    'ViT-L/14': 'CLIP_VIT_L14',
    'ViT-B/32': 'ViT-B_32',
}

model_list   = args.target_models
_CLIP_MODELS = {'ViT-B/16', 'ViT-B/32', 'ViT-L/14', 'RN50', 'RN50x4'}
_BERT_MODELS = {'ALBEF', 'TCL', 'BLIP', 'XVLM'}

# ==========================================
# AWGN channel
# ==========================================

def parse_snr(snr_str):
    return float('inf') if snr_str.lower() == 'inf' else float(snr_str)


def apply_awgn(feat, snr_db):
    """Corrupt `feat` with AWGN at `snr_db` (Eqs. (9)-(10)).

    The noise power is derived from the mean signal power of the batch and the
    result is re-normalized along the last dimension. Applied to the transmitted
    embeddings and, with --gallery_awgn pixel (paper runs), also to the
    UAP-perturbed gallery images, whose last dimension is the image width.
    """
    if snr_db == float('inf'):
        return feat
    p_signal   = (feat ** 2).mean().item()
    snr_linear = 10 ** (snr_db / 10.0)
    sigma      = np.sqrt(max(p_signal, 1e-8) / snr_linear)
    noise = torch.randn_like(feat) * sigma
    return F.normalize(feat + noise, dim=-1)


def get_ckpt(model_name, config):
    if model_name in _CLIP_MODELS:
        return ''
    return os.path.join(args.checkpoint, model_name, f'{config["dataset"]}.pth')


def load_one_model(model_name, config):
    """Load a single evaluation model onto CPU."""
    ckpt = get_ckpt(model_name, config)
    cpu  = torch.device('cpu')
    if model_name == 'BLIP':
        model, ref_model, blip_tok = load_model(
            model_name, ckpt, args.text_encoder, config, cpu)
        return model, ref_model, None, blip_tok
    else:
        model, ref_model, tok = load_model(
            model_name, ckpt, args.text_encoder, config, cpu)
        return model, ref_model, tok, None


# ==========================================
# uap_embedding -> adv_words (nearest-neighbour word lookup)
# ==========================================

def embedding_to_adv_words(uap_embeddings, tokenizer, device):
    """
    Map the text UAP (K, D) to the K nearest vocabulary words.
    BERT-family encoders use word_idx_{source_model}.npy, CLIP uses
    word_idx_CLIP.npy.
    """
    is_clip = args.source_model in _CLIP_MODELS
    if is_clip:
        word_idxs       = np.load('word_idx_CLIP.npy')
        word_embeddings = torch.load('word_embedding_CLIP.pth',
                                     map_location='cpu').to(device)
    else:
        word_idxs       = np.load(f'word_idx_{args.source_model}.npy')
        word_embeddings = torch.load(f'word_embedding_{args.source_model}.pth',
                                     map_location='cpu').to(device)

    available_we = word_embeddings[word_idxs]
    adv_words    = []

    for i in range(uap_embeddings.size(0)):
        uap_emb    = uap_embeddings[i].to(device)
        similarity = torch.stack([torch.dist(uap_emb, we, p=2)
                                   for we in available_we])
        for k in range(similarity.size(0)):
            _, indices = torch.topk(similarity, k=k + 1, largest=False, sorted=True)
            min_id = word_idxs[indices[k]]
            if is_clip:
                word = SimpleTokenizer().decoder[min_id].replace('</w>', '')
            else:
                word = tokenizer._convert_id_to_token(min_id)
            if word not in filter_words:
                adv_words.append(word)
                break

    print(f'adv_words: {adv_words}')
    return adv_words


# ==========================================
# Gallery feature extraction (clean and adversarial image gallery)
# ==========================================
def extract_gallery_feats(tokenizer, blip_tokenizer, target_transform,
                           data_loader, device, config,
                           uap_noise=None, snr_db=None):
    """Extract gallery features.

    uap_noise: if given, the gallery images are perturbed with the UAP and
               transmitted over the AWGN channel, in pixel or embedding space
               (--gallery_awgn): the adversarial image gallery used for T2I.
    snr_db:    channel SNR used together with uap_noise.
    """
    mode_tag = 'adv_img' if uap_noise is not None else 'clean'
    print(f'\n[Gallery:{mode_tag}] Extracting...')
    images_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
    num_text  = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)
    gallery   = {}
    for model_name in model_list:
        print(f'  Loading {model_name}...')
        model, _, tok, blip_tok = load_one_model(model_name, config)
        model = model.to(device)
        model.eval()
        out_dim = model.visual.output_dim if model_name in _CLIP_MODELS \
                  else config['embed_dim']
        gallery[model_name] = {
            'image_feats': torch.zeros(num_image, out_dim),
            'text_feats':  torch.zeros(num_text,  out_dim),
        }
        tgt_is_vlp = model_name in _BERT_MODELS
        _tok       = blip_tokenizer if model_name == 'BLIP' else tokenizer
        for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(
                tqdm(data_loader, desc=f'  Gallery [{model_name}]')):
            texts_ids = []
            texts     = []
            for i in range(len(texts_group)):
                texts     += texts_group[i]
                texts_ids += text_ids_groups[i]
            images = images.to(device)
            with torch.no_grad():
                # Adversarial image gallery: apply the UAP. With
                # --gallery_awgn pixel (paper runs) the channel noise is added
                # to the perturbed images here, before encoding.
                if uap_noise is not None:
                    eval_images = torch.clamp(
                        images + uap_noise[0:1].expand(images.size()), 0.0, 1.0)
                    if args.gallery_awgn == 'pixel':
                        eval_images = apply_awgn(eval_images, snr_db)
                else:
                    eval_images = images
                t_imgs = torch.stack(
                    [target_transform(img) for img in eval_images], dim=0).to(device)
                src_is_vlp = args.source_model in _BERT_MODELS
                cur_images = (eval_images if src_is_vlp else t_imgs) if tgt_is_vlp \
                             else (t_imgs if src_is_vlp else eval_images)
                cur_norm   = images_normalize(cur_images)
                if tgt_is_vlp:
                    ti = _tok(texts, padding='max_length', truncation=True,
                              max_length=30, return_tensors="pt").to(device)
                    out_img  = model.inference_image(cur_norm)
                    out_txt  = model.inference_text(ti)
                    img_feat = out_img['image_feat'].cpu().detach().float()
                    txt_feat = out_txt['text_feat'].cpu().detach().float()
                    del ti, out_img, out_txt
                else:
                    output   = model.inference(cur_norm, texts)
                    img_feat = output['image_feat'].cpu().float().detach()
                    txt_feat = output['text_feat'].cpu().float().detach()
                    del output

                # --gallery_awgn embedding: the encoded gallery embeddings are
                # transmitted over the channel instead.
                if uap_noise is not None and args.gallery_awgn == 'embedding':
                    img_feat = apply_awgn(img_feat, snr_db)

                gallery[model_name]['image_feats'][images_ids] = img_feat
                gallery[model_name]['text_feats'][texts_ids]   = txt_feat
                del t_imgs, cur_norm, cur_images
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print(f'  {model_name} gallery done.')
    return gallery


def extract_query_feats(tokenizer, blip_tokenizer, target_transform,
                         data_loader, device, config,
                         snr_db, uap_noise=None, txt_attacker=None,
                         is_adv=False):
    """Extract query features.

    is_adv=False: clean query.
    is_adv=True:  adversarial query -- the image carries the UAP and the text
                  is perturbed by TextAttacker token substitution.
    Both modalities are then transmitted over the AWGN channel (embedding
    space).
    """
    images_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))

    mode_tag = 'adv' if is_adv else 'clean'
    snr_tag  = 'inf' if snr_db == float('inf') else f'{snr_db:.0f}dB'
    print(f'  Extracting query [{mode_tag}] [SNR={snr_tag}]...')

    num_text  = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)
    query     = {}
    cached_adv_texts = {}  # computed on the source model, reused for others

    for model_name in model_list:
        print(f'    Loading {model_name}...')
        model, _, tok, blip_tok = load_one_model(model_name, config)
        model = model.to(device)
        model.eval()

        out_dim = model.visual.output_dim if model_name in _CLIP_MODELS \
                  else config['embed_dim']
        query[model_name] = {
            'image_feats': torch.zeros(num_image, out_dim),
            'text_feats':  torch.zeros(num_text,  out_dim),
        }

        tgt_is_vlp = model_name in _BERT_MODELS
        _tok       = blip_tokenizer if model_name == 'BLIP' else tokenizer

        for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(
                tqdm(data_loader, desc=f'    {model_name}')):
            texts_ids = []
            texts     = []
            for i in range(len(texts_group)):
                texts     += texts_group[i]
                texts_ids += text_ids_groups[i]

            images = images.to(device)

            # Image: adversarial queries carry the UAP.
            if is_adv and uap_noise is not None:
                eval_images = torch.clamp(
                    images + uap_noise[0:1].expand(images.size()), 0.0, 1.0)
            else:
                eval_images = images

            # Text: adversarial queries use TextAttacker token substitution,
            # computed when the loop reaches the source model and reused for
            # the targets after it. Targets listed before args.source_model
            # fall back to the clean captions; this order dependence is that of
            # the paper runs (target order ALBEF, TCL, BLIP, XVLM, ViT-B/16,
            # ViT-B/32) and is kept for parity.
            if is_adv and txt_attacker is not None:
                if model_name == args.source_model:
                    eval_texts = txt_attacker.get_adv_text(model, texts)
                    cached_adv_texts[batch_idx] = eval_texts
                else:
                    eval_texts = cached_adv_texts.get(batch_idx, texts)
            else:
                eval_texts = texts

            with torch.no_grad():
                t_imgs = torch.stack(
                    [target_transform(img) for img in eval_images], dim=0).to(device)

                src_is_vlp = args.source_model in _BERT_MODELS
                cur_images = (eval_images if src_is_vlp else t_imgs) if tgt_is_vlp \
                             else (t_imgs if src_is_vlp else eval_images)
                cur_norm   = images_normalize(cur_images)

                if tgt_is_vlp:
                    ti = _tok(eval_texts, padding='max_length', truncation=True,
                              max_length=30, return_tensors="pt").to(device)
                    out_img  = model.inference_image(cur_norm)
                    out_txt  = model.inference_text(ti)
                    img_feat = out_img['image_feat'].cpu().detach().float()
                    txt_feat = out_txt['text_feat'].cpu().detach().float()
                    del ti, out_img, out_txt
                else:
                    output   = model.inference(cur_norm, eval_texts)
                    img_feat = output['image_feat'].cpu().float().detach()
                    txt_feat = output['text_feat'].cpu().float().detach()
                    del output

                # Transmission over the AWGN channel (embedding space).
                img_feat = apply_awgn(img_feat, snr_db)
                txt_feat = apply_awgn(txt_feat, snr_db)

                query[model_name]['image_feats'][images_ids] = img_feat
                query[model_name]['text_feats'][texts_ids]   = txt_feat

                del t_imgs, cur_norm, cur_images

        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print(f'    {model_name} done.')

    return query


# ==========================================
# Score matrices (cosine similarity on normalized embeddings)
# ==========================================

def compute_score_matrix(query_feats, clean_gallery_feats, adv_img_gallery_feats=None):
    """
    i2t: adv image query  @ clean text gallery
    t2i: adversarial text query @ adversarial image gallery when provided,
         otherwise the clean image gallery
    """
    scores = {}
    for model_name in model_list:
        q_img = query_feats[model_name]['image_feats']
        g_txt = clean_gallery_feats[model_name]['text_feats']
        q_txt = query_feats[model_name]['text_feats']
        # prefer the adversarial image gallery for T2I
        if adv_img_gallery_feats is not None:
            g_img = adv_img_gallery_feats[model_name]['image_feats']
        else:
            g_img = clean_gallery_feats[model_name]['image_feats']
        scores[model_name] = {
            'i2t': (q_img @ g_txt.t()).numpy(),
            't2i': (q_txt @ g_img.t()).numpy(),
        }
    return scores


# ==========================================
# Attack success rates
# ==========================================

def compute_asr(clean_scores, adv_scores, data_loader,
                inf_adv_ranks=None):
    """
    ASR(gamma)           = |clean_success(gamma) & adv_fail(gamma)| / |clean_success(gamma)|
    Pure UAP ASR(gamma)  = |clean_success(gamma) & adv_fail(inf)|   / |clean_success(gamma)|

    Pure UAP ASR (Eq. (23)) fixes the failure set at gamma = inf, isolating the
    standalone contribution of the UAP from the channel noise.
    """
    img2txt = data_loader.dataset.img2txt
    txt2img = data_loader.dataset.txt2img
    results = {}

    for model_name in model_list:
        c_i2t = clean_scores[model_name]['i2t']
        a_i2t = adv_scores[model_name]['i2t']
        clean_ranks_i2t = np.zeros(c_i2t.shape[0])
        adv_ranks_i2t   = np.zeros(a_i2t.shape[0])
        for idx in range(c_i2t.shape[0]):
            c_inds = np.argsort(c_i2t[idx])[::-1]
            a_inds = np.argsort(a_i2t[idx])[::-1]
            c_rank = min(np.where(c_inds == gt)[0][0] for gt in img2txt[idx])
            a_rank = min(np.where(a_inds == gt)[0][0] for gt in img2txt[idx])
            clean_ranks_i2t[idx] = c_rank
            adv_ranks_i2t[idx]   = a_rank
        c_t2i = clean_scores[model_name]['t2i']
        a_t2i = adv_scores[model_name]['t2i']
        clean_ranks_t2i = np.zeros(c_t2i.shape[0])
        adv_ranks_t2i   = np.zeros(a_t2i.shape[0])
        for idx in range(c_t2i.shape[0]):
            c_inds = np.argsort(c_t2i[idx])[::-1]
            a_inds = np.argsort(a_t2i[idx])[::-1]
            clean_ranks_t2i[idx] = np.where(c_inds == txt2img[idx])[0][0]
            adv_ranks_t2i[idx]   = np.where(a_inds == txt2img[idx])[0][0]

        def asr_at_k(c_ranks, a_ranks, k):
            c_ok   = set(np.where(c_ranks < k)[0])
            a_fail = set(np.where(a_ranks >= k)[0])
            return round(100.0 * len(c_ok & a_fail) / max(len(c_ok), 1), 2)

        def pure_uap_asr_at_k(c_ranks, inf_a_ranks, k):
            c_ok       = set(np.where(c_ranks    < k)[0])
            a_fail_inf = set(np.where(inf_a_ranks >= k)[0])
            return round(100.0 * len(c_ok & a_fail_inf) / max(len(c_ok), 1), 2)

        r = {
            'asr_tr1':   asr_at_k(clean_ranks_i2t, adv_ranks_i2t, 1),
            'asr_tr5':   asr_at_k(clean_ranks_i2t, adv_ranks_i2t, 5),
            'asr_tr10':  asr_at_k(clean_ranks_i2t, adv_ranks_i2t, 10),
            'asr_ir1':   asr_at_k(clean_ranks_t2i, adv_ranks_t2i, 1),
            'asr_ir5':   asr_at_k(clean_ranks_t2i, adv_ranks_t2i, 5),
            'asr_ir10':  asr_at_k(clean_ranks_t2i, adv_ranks_t2i, 10),
            'clean_tr1': round(100.0 * np.mean(clean_ranks_i2t < 1), 2),
            'clean_ir1': round(100.0 * np.mean(clean_ranks_t2i < 1), 2),
            'adv_tr1':   round(100.0 * np.mean(adv_ranks_i2t   < 1), 2),
            'adv_ir1':   round(100.0 * np.mean(adv_ranks_t2i   < 1), 2),
            '_adv_ranks_i2t': adv_ranks_i2t,
            '_adv_ranks_t2i': adv_ranks_t2i,
        }

        if inf_adv_ranks is not None and model_name in inf_adv_ranks:
            inf_ai2t = inf_adv_ranks[model_name]['i2t']
            inf_at2i = inf_adv_ranks[model_name]['t2i']
            r['pure_asr_tr1']  = pure_uap_asr_at_k(clean_ranks_i2t, inf_ai2t, 1)
            r['pure_asr_tr5']  = pure_uap_asr_at_k(clean_ranks_i2t, inf_ai2t, 5)
            r['pure_asr_tr10'] = pure_uap_asr_at_k(clean_ranks_i2t, inf_ai2t, 10)
            r['pure_asr_ir1']  = pure_uap_asr_at_k(clean_ranks_t2i, inf_at2i, 1)
            r['pure_asr_ir5']  = pure_uap_asr_at_k(clean_ranks_t2i, inf_at2i, 5)
            r['pure_asr_ir10'] = pure_uap_asr_at_k(clean_ranks_t2i, inf_at2i, 10)
        else:
            for key in ['pure_asr_tr1','pure_asr_tr5','pure_asr_tr10',
                        'pure_asr_ir1','pure_asr_ir5','pure_asr_ir10']:
                r[key] = None

        results[model_name] = r

        pure_tr = f' pure={r["pure_asr_tr1"]:.2f}' if r['pure_asr_tr1'] is not None else ''
        pure_ir = f' pure={r["pure_asr_ir1"]:.2f}' if r['pure_asr_ir1'] is not None else ''
        print(f'  {model_name}: '
              f'i2t clean={r["clean_tr1"]:.2f} adv={r["adv_tr1"]:.2f} '
              f'ASR={r["asr_tr1"]:.2f}{pure_tr} | '
              f't2i clean={r["clean_ir1"]:.2f} adv={r["adv_ir1"]:.2f} '
              f'ASR={r["asr_ir1"]:.2f}{pure_ir}')
    return results

# ==========================================
# Entry point
# ==========================================

yaml_loader = YAML(typ='safe')
config      = yaml_loader.load(open(args.config, 'r'))
device      = torch.device('cuda')

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

# Only the tokenizer and ref_model are needed here (for TextAttacker)
print("Loading tokenizer and ref_model...")
_, ref_model, _tok, _blip_tok = load_one_model(args.source_model, config)
ref_model = ref_model.cpu()
# BLIP returns a blip_tokenizer, the other models a plain tokenizer
tokenizer = _tok if _tok is not None else _blip_tok
gc.collect()

blip_tokenizer = None
if 'BLIP' in model_list:
    _, _, _, blip_tokenizer = load_one_model('BLIP', config)
    gc.collect()

print("Creating dataset...")
if args.source_model in _BERT_MODELS:
    source_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
else:
    n_px = 224
    source_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor()])

test_dataset = paired_dataset(
    config['annotation_file'], source_transform, config['image_root'])
test_loader  = DataLoader(
    test_dataset, batch_size=args.batch_size,
    num_workers=0, collate_fn=test_dataset.collate_fn)

# Load the exported UAP
src_tag  = _EVAL_TAG_MAP.get(args.source_model, args.source_model)
load_dir = os.path.join(args.load_dir, src_tag, config['dataset'])

uap_noise_path = os.path.join(load_dir, f'uap_noise-{args.epoch}.pth')
if not os.path.exists(uap_noise_path):
    raise FileNotFoundError(f'uap_noise not found: {uap_noise_path}')
uap_noise = torch.load(uap_noise_path, map_location='cpu')

# Text attacker: uap_embedding -> adv_words -> TextAttacker
txt_attacker = None
uap_embedding_path = os.path.join(load_dir, f'uap_embedding-{args.epoch}.pth')
if os.path.exists(uap_embedding_path):
    print("Found uap_embedding, building TextAttacker...")
    uap_embeddings = torch.load(uap_embedding_path, map_location='cpu')
    # exported shape is (B, D), one row per batch sample of the last training
    # iteration; it is averaged into a single text UAP, i.e. one adversarial
    # word, as in the paper runs
    if uap_embeddings.dim() == 2 and uap_embeddings.shape[0] > 1:
        uap_embeddings = uap_embeddings.mean(dim=0, keepdim=True)
    adv_words    = embedding_to_adv_words(uap_embeddings, tokenizer, device)
    ref_model    = ref_model.to(device)  # the attacker needs it on GPU
    txt_attacker = TextAttacker(
        ref_model, tokenizer, cls=False, max_length=30,
        number_perturbation=args.text_budget, adv_words=adv_words,
        topk=10, threshold_pred_score=0.3)
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("No uap_embedding found, image-only attack mode.")

snr_values = [parse_snr(s) for s in args.snr_list]
print(f"\nTesting SNR: {['inf' if s==float('inf') else f'{s:.0f}dB' for s in snr_values]}")

# Step 1: clean gallery, extracted once (provides the I2T text gallery)
gallery_feats = extract_gallery_feats(
    tokenizer, blip_tokenizer, target_transform,
    test_loader, device, config)

# Step 2: per SNR, run the clean and adversarial queries and score them
all_results   = {}
inf_adv_ranks = None  # adversarial ranks at gamma = inf, for Pure UAP ASR
uap_noise_gpu = uap_noise.to(device)

for snr_db in snr_values:
    snr_tag = 'inf' if snr_db == float('inf') else f'{snr_db:.0f}dB'
    print(f'\n{"="*60}')
    print(f'  SNR = {snr_tag}')
    print(f'{"="*60}')

    clean_query = extract_query_feats(
        tokenizer, blip_tokenizer, target_transform,
        test_loader, device, config,
        snr_db, uap_noise=None, txt_attacker=None, is_adv=False)
    clean_scores = compute_score_matrix(clean_query, gallery_feats)

    adv_query = extract_query_feats(
        tokenizer, blip_tokenizer, target_transform,
        test_loader, device, config,
        snr_db, uap_noise=uap_noise_gpu, txt_attacker=txt_attacker, is_adv=True)

    # adversarial image gallery for T2I (adv text query @ adv image gallery)
    adv_img_gallery = extract_gallery_feats(
        tokenizer, blip_tokenizer, target_transform,
        test_loader, device, config,
        uap_noise=uap_noise_gpu, snr_db=snr_db)

    adv_scores = compute_score_matrix(adv_query, gallery_feats, adv_img_gallery)

    print(f'\n[ASR @ SNR={snr_tag}]')
    snr_results = compute_asr(clean_scores, adv_scores, test_loader,
                              inf_adv_ranks=inf_adv_ranks)
    all_results[snr_tag] = snr_results

    # cache the noiseless adversarial ranks for the Pure UAP ASR denominator
    if snr_db == float('inf'):
        inf_adv_ranks = {}
        for mn in model_list:
            inf_adv_ranks[mn] = {
                'i2t': snr_results[mn]['_adv_ranks_i2t'].copy(),
                't2i': snr_results[mn]['_adv_ranks_t2i'].copy(),
            }

    del clean_query, adv_query, clean_scores, adv_scores
    torch.cuda.empty_cache()
    gc.collect()

uap_noise_gpu = uap_noise_gpu.cpu()

# Summary table
print(f'\n{"="*75}')
print(f'Summary: {args.source_model} - ASR vs SNR')
print(f'{"="*75}')
print(f'{"SNR":<8} {"i2t_clean":<12} {"i2t_adv":<12} {"i2t_ASR":<12} '
      f'{"t2i_clean":<12} {"t2i_adv":<12} {"t2i_ASR":<12}')
print('-' * 75)
for snr_tag, results in all_results.items():
    r = results.get(args.source_model, {})
    if r:
        print(f'{snr_tag:<8} {r["clean_tr1"]:<12.2f} {r["adv_tr1"]:<12.2f} '
              f'{r["asr_tr1"]:<12.2f} {r["clean_ir1"]:<12.2f} '
              f'{r["adv_ir1"]:<12.2f} {r["asr_ir1"]:<12.2f}')

# Write the results to disk
save_dir  = f'./outputs/{args.output}'
os.makedirs(save_dir, exist_ok=True)
xlsx_path = os.path.join(
    save_dir,
    f"{args.source_model.replace('/', '_')}_{config['dataset']}_asr.xlsx")

with pd.ExcelWriter(xlsx_path) as writer:
    for snr_tag, results in all_results.items():
        rows = []
        for model_name in model_list:
            r = results.get(model_name, {})
            if r:
                rows.append({
                    'Model':           model_name,
                    'i2t_clean':       r['clean_tr1'], 'i2t_adv':       r['adv_tr1'],
                    'i2t_ASR_r1':      r['asr_tr1'],   'i2t_ASR_r5':    r['asr_tr5'],
                    'i2t_ASR_r10':     r['asr_tr10'],
                    'i2t_pureASR_r1':  r['pure_asr_tr1'],
                    'i2t_pureASR_r5':  r['pure_asr_tr5'],
                    'i2t_pureASR_r10': r['pure_asr_tr10'],
                    't2i_clean':       r['clean_ir1'], 't2i_adv':       r['adv_ir1'],
                    't2i_ASR_r1':      r['asr_ir1'],   't2i_ASR_r5':    r['asr_ir5'],
                    't2i_ASR_r10':     r['asr_ir10'],
                    't2i_pureASR_r1':  r['pure_asr_ir1'],
                    't2i_pureASR_r5':  r['pure_asr_ir5'],
                    't2i_pureASR_r10': r['pure_asr_ir10'],
                })
        pd.DataFrame(rows).to_excel(writer, sheet_name=f'SNR_{snr_tag}', index=False)

    summary_rows = []
    for snr_tag, results in all_results.items():
        for model_name in model_list:
            r = results.get(model_name, {})
            if r:
                summary_rows.append({
                    'SNR':        snr_tag,
                    'Model':      model_name,
                    'i2t_clean':  r['clean_tr1'], 'i2t_adv':  r['adv_tr1'],
                    'i2t_ASR_r1': r['asr_tr1'],
                    't2i_clean':  r['clean_ir1'], 't2i_adv':  r['adv_ir1'],
                    't2i_ASR_r1': r['asr_ir1'],
                })
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

print(f'\nSaved to {xlsx_path}')
