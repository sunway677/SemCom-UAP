"""
Channel-aware UAP attackers for SemCom-UAP.

Implements Sections IV-C to IV-E of the paper:
  * Conditional UAP generation             -- Eqs. (3)-(6)
  * AWGN corruption in the embedding space -- Eqs. (7)-(10)
  * Adversarial cross-modal contrastive (ACMC) learning -- Eqs. (11)-(14)
  * Stratified SNR sampling                -- Eq. (20)

Each attacker owns two Adam optimizers: one over its generator and the shared
SNR encoder f_snr, and one over its latent code z (learning rate scaled by
z_lr_scale). `attack()` runs the primary-surrogate branch over the stratified
SNR samples, back-propagates the stratum-averaged loss and steps both
optimizers. The auxiliary-surrogate update is a separate step taken by the
training script (train_semcom_uap.py).
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


CLIP_ENCODERS = {'ViT-B/32', 'ViT-B/16'}

ENCODER_FEAT_DIM = {
    'ALBEF':    256,
    'TCL':      256,
    'XVLM':     256,
    'BLIP':     256,
    'ViT-B/32': 512,
    'ViT-B/16': 512,
}

# Per-surrogate prior used to initialize the UCB running mean of the raw
# auxiliary loss (running_mean = 3 * difficulty; see AuxiliaryUCB).
ENCODER_DIFFICULTY = {
    'ALBEF':    0.30,
    'TCL':      0.30,
    'BLIP':     0.35,
    'XVLM':     0.40,
    'ViT-B/32': 0.50,
    'ViT-B/16': 0.60,
}

SNR_EMB_DIM = 64


# ==========================================
# AWGN channel (Eqs. (9)-(10))
# ==========================================

class AWGNChannel:
    """Embedding-space AWGN channel.

    The noise standard deviation is shared across the batch and derived from
    the batch-averaged signal power, estimated without gradient:

        sigma_n  = sqrt(p_signal / 10^(gamma/10))
        p_signal = (1 / (d * |B|)) * sum_i ||v_i^adv||_2^2

    For gamma >= 1000 dB (and gamma = inf) the noise is exactly zero; for such
    SNRs the formula above already underflows to zero noise in float32.
    """

    def __init__(self, snr_min_db=0.0, snr_max_db=20.0):
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    @staticmethod
    def sigma(embedding, snr_db):
        if np.isinf(snr_db) or snr_db >= 1000:
            return 0.0
        p_signal = (embedding.detach() ** 2).mean().item()
        return float(np.sqrt(max(p_signal, 1e-8) / (10 ** (snr_db / 10.0))))

    def __call__(self, embedding, snr_db=None):
        """Corrupt `embedding` with AWGN at `snr_db`, then re-normalize."""
        if snr_db is None:
            snr_db = np.random.uniform(self.snr_min_db, self.snr_max_db)
        sigma = self.sigma(embedding, snr_db)
        noise = torch.randn_like(embedding) * sigma
        return F.normalize(embedding + noise, dim=-1), snr_db


def stratified_snr_sample(snr_min, snr_max, num_strata=3):
    """Stratified SNR sampling (Eq. (20)): one draw per equal-width stratum."""
    seg = (snr_max - snr_min) / num_strata
    return [np.random.uniform(snr_min + m * seg, snr_min + (m + 1) * seg)
            for m in range(num_strata)]


# ==========================================
# Adversarial cross-modal contrastive loss (Eqs. (11)-(14))
# ==========================================

def select_farthest_target(query_feats, clean_img_feats, clean_txt_feats):
    """Farthest cross-modal target selection (Eq. (11)).

        y_i = argmin_{j != i} ( q_i^T v_j + q_i^T u_j )

    Returns the clean image/text embeddings at index y_i.
    """
    with torch.no_grad():
        q = F.normalize(query_feats.detach(), dim=-1)
        v = F.normalize(clean_img_feats.detach(), dim=-1)
        u = F.normalize(clean_txt_feats.detach(), dim=-1)
        sim  = q @ v.T + q @ u.T
        mask = torch.eye(q.shape[0], device=q.device).bool()
        sim  = sim.masked_fill(mask, float('inf'))
        idx  = sim.argmin(dim=1)
    return v[idx], u[idx]


def acmc_loss(adv_feats, target_img_feats, target_txt_feats,
              clean_img_feats, clean_txt_feats,
              temperature=0.07, margin=0.02):
    """Margin-based adversarial cross-modal contrastive loss (Eq. (12)).

    The loss averages an image-target term and a text-target term. The
    partition function of the image-target term aggregates the selected image
    target, the sample's own clean image feature and all other clean image
    features of the batch; the text-target term does the same with the clean
    text features. The same loss is used by the image and text branches and by
    the auxiliary surrogates.
    """
    B    = adv_feats.shape[0]
    adv  = F.normalize(adv_feats, dim=-1)
    ml   = margin / temperature
    mask = 1.0 - torch.eye(B, device=adv.device)

    # image target, image negatives
    l_pos_i  = (adv * target_img_feats).sum(-1) / temperature
    l_self_i = (adv * clean_img_feats).sum(-1) / temperature
    l_neg_i  = (adv @ clean_img_feats.T) / temperature
    z_i      = torch.exp(l_pos_i) + torch.exp(l_self_i) + (torch.exp(l_neg_i) * mask).sum(-1)
    l_img    = -(l_pos_i - ml) + torch.log(z_i + 1e-8)

    # text target, text negatives
    l_pos_t  = (adv * target_txt_feats).sum(-1) / temperature
    l_self_t = (adv * clean_txt_feats).sum(-1) / temperature
    l_neg_t  = (adv @ clean_txt_feats.T) / temperature
    z_t      = torch.exp(l_pos_t) + torch.exp(l_self_t) + (torch.exp(l_neg_t) * mask).sum(-1)
    l_txt    = -(l_pos_t - ml) + torch.log(z_t + 1e-8)

    return ((l_img + l_txt) / 2.0).mean()


# ==========================================
# Image branch
# ==========================================

class ChannelAwareImageAttacker:
    """SNR-aware image perturbation generator wrapper.

    The generator emits one perturbation row per latent row (one per sample of
    the batch). Row i perturbs image i during training; the per-sample tensor
    returned by the last attack() call of an epoch is what the training script
    exports.
    """

    def __init__(self, generator, snr_encoder, z_init,
                 eps=12 / 255, device='cuda', lr=2e-4,
                 temperature=0.07, z_lr_scale=0.05,
                 snr_min_db=0.0, snr_max_db=20.0,
                 margin=0.02, num_snr_strata=3):
        self.eps            = eps
        self.generator      = generator.to(device)
        self.snr_encoder    = snr_encoder.to(device)
        self.device         = device
        self.temperature    = temperature
        self.margin         = margin
        self.num_snr_strata = num_snr_strata
        self.snr_min        = snr_min_db
        self.snr_max        = snr_max_db
        self.z              = z_init
        self.channel        = AWGNChannel(snr_min_db, snr_max_db)

        # G_x and the shared f_snr; the text attacker keeps its own Adam state
        # for f_snr.
        params = list(self.generator.parameters()) + list(self.snr_encoder.parameters())
        self.optimG = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        self.z.requires_grad_(True)
        self.optimZ = optim.Adam([self.z], lr=lr * z_lr_scale, betas=(0.5, 0.999))

    # ---- conditioning (Eq. (3)) --------------------------------------
    def build_cond(self, ref_feats, snr_db):
        """c_{x,i} = [ e_i ; f_snr(gamma) ] (one condition row per sample)."""
        B       = ref_feats.shape[0]
        snr_emb = self.snr_encoder(snr_db, self.device).expand(B, -1)
        return torch.cat([ref_feats, snr_emb], dim=-1)

    # ---- UAP generation (Eq. (5) + clipping) -------------------------
    def generate_uap(self, ref_feats, snr_db, size=None):
        """Per-sample perturbations clipped to the l_inf budget, (B, 3, H, W).

        If `size` is given (CLIP primary), the raw generator output is first
        resized to size x size and then clipped.
        """
        cond = self.build_cond(ref_feats, snr_db)
        uap  = self.generator(self.z, cond)
        if size is not None:
            uap = F.interpolate(uap, size=(size, size),
                                mode='bilinear', align_corners=False)
        while uap.dim() > 4:
            uap = uap.squeeze(1)
        if uap.dim() == 3:
            uap = uap.unsqueeze(0)
        return torch.clamp(uap, -self.eps, self.eps)

    @staticmethod
    def apply_uap(images, uap, model=None, is_clip=False):
        """x^adv = clip(x + delta_x, 0, 1), resizing for CLIP if needed."""
        if is_clip:
            size = model.visual.input_resolution
            if images.shape[-1] != size:
                images = F.interpolate(images, size=(size, size),
                                       mode='bilinear', align_corners=False)
            if uap.shape[-2:] != (size, size):
                uap = F.interpolate(uap, size=(size, size),
                                    mode='bilinear', align_corners=False)
        return torch.clamp(images + uap.expand_as(images), 0.0, 1.0)

    @staticmethod
    def encode(model, imgs_norm, is_clip):
        feat = model.encode_image(imgs_norm) if is_clip \
               else model.inference_image(imgs_norm)['image_feat']
        return F.normalize(feat, dim=-1)

    # ---- primary update (Eqs. (7)-(12), (21)) ------------------------
    def attack(self, model, enc_name, normalization,
               images, img2txt, ref_txt_feats,
               clean_img_feats, clean_txt_feats):
        """One image-branch update on the primary surrogate.

        images:          (B, 3, H, W) clean images in [0, 1]
        img2txt:         caption indices of each image
        ref_txt_feats:   (N_txt, D) caption-level reference features; the
                         condition of image i is their mean over img2txt[i]
        clean_*_feats:   (B, D) clean features from the feature bank

        The clean images are augmented with U(-eps_x, eps_x) noise, M SNR values
        are drawn by stratified sampling, and the ACMC loss on the
        AWGN-corrupted adversarial embeddings is averaged over the strata
        before one optimizer step.

        Returns (loss, last_uap), where last_uap is the per-sample
        perturbation of the last stratum (resized to the CLIP input resolution
        for CLIP primaries).
        """
        model.eval()
        is_clip = enc_name in CLIP_ENCODERS

        aug_imgs = torch.clamp(
            images.detach() + torch.from_numpy(
                np.random.uniform(-self.eps, self.eps, images.shape)
            ).float().to(self.device),
            0.0, 1.0)

        self.generator.train()
        self.snr_encoder.train()
        self.optimG.zero_grad()
        self.optimZ.zero_grad()

        x = aug_imgs.to(self.device)

        snr_list = stratified_snr_sample(self.snr_min, self.snr_max,
                                         self.num_snr_strata)

        clean_v = F.normalize(clean_img_feats.to(self.device), dim=-1)
        clean_u = F.normalize(clean_txt_feats.to(self.device), dim=-1)

        base_cond = torch.stack([
            torch.mean(ref_txt_feats[img2txt[i]], dim=0)
            for i in range(len(img2txt))
        ], dim=0)

        losses   = []
        last_uap = None

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for snr_db in snr_list:
                if is_clip:
                    size = model.visual.input_resolution
                    xi   = F.interpolate(x, size=(size, size),
                                         mode='bilinear', align_corners=False) \
                           if x.shape[-1] != size else x
                    uap  = self.generate_uap(base_cond, snr_db, size=size)
                else:
                    xi   = x
                    uap  = self.generate_uap(base_cond, snr_db)
                last_uap = uap

                adv_imgs  = torch.clamp(xi + uap.expand_as(xi), 0.0, 1.0)
                adv_feats = self.encode(model, normalization(adv_imgs), is_clip)

                noisy, _     = self.channel(adv_feats, snr_db)
                t_img, t_txt = select_farthest_target(noisy, clean_v, clean_u)
                losses.append(acmc_loss(noisy, t_img, t_txt, clean_v, clean_u,
                                        temperature=self.temperature,
                                        margin=self.margin))

            total_loss = torch.stack(losses).mean()

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimG.step()
            self.optimZ.step()

        return total_loss.detach(), last_uap

    # ---- checkpointing ------------------------------------------------
    def save_model(self, path):
        torch.save({'generator':   self.generator.state_dict(),
                    'snr_encoder': self.snr_encoder.state_dict()}, path)

    def save_z(self, path):
        torch.save(self.z.data, path)

    def load_model(self, path, device='cuda'):
        d = torch.load(path, map_location=device)
        self.generator.load_state_dict(d['generator'])
        self.snr_encoder.load_state_dict(d['snr_encoder'])

    def load_z(self, path, device='cuda'):
        self.z.data = torch.load(path, map_location=device)


# ==========================================
# Text branch
# ==========================================

class ChannelAwareTextAttacker:
    """SNR-aware text embedding perturbation generator wrapper.

    Latent row i is conditioned on the feature of image i. Every row is
    norm-clamped to the natural word-embedding magnitude range; the first
    `text_budget` rows are substituted into the captions during training, and
    all rows returned by the last attack() call of an epoch are exported.
    """

    def __init__(self, ref_model_name, generator, snr_encoder, z_init,
                 device='cuda', temperature=0.07,
                 min_norm=None, max_norm=None,
                 lr=2e-4, max_length=30,
                 text_budget=1, z_lr_scale=0.05,
                 snr_min_db=0.0, snr_max_db=20.0,
                 margin=0.02, num_snr_strata=3):
        self.ref_model_name = ref_model_name
        self.device         = device
        self.temperature    = temperature
        self.margin         = margin
        self.min_norm       = min_norm
        self.max_norm       = max_norm
        self.max_length     = max_length
        self.text_budget    = text_budget
        self.num_snr_strata = num_snr_strata
        self.snr_min        = snr_min_db
        self.snr_max        = snr_max_db
        self.channel        = AWGNChannel(snr_min_db, snr_max_db)

        self.generator   = generator.to(device)
        self.snr_encoder = snr_encoder.to(device)
        self.z           = z_init

        # G_t and the shared f_snr (separate Adam state from the image attacker)
        params = list(self.generator.parameters()) + list(self.snr_encoder.parameters())
        self.optimG = optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        self.z.requires_grad_(True)
        self.optimZ = optim.Adam([self.z], lr=lr * z_lr_scale, betas=(0.5, 0.999))

    # ---- conditioning (Eq. (4)) --------------------------------------
    def build_cond(self, ref_feats, snr_db):
        """c_{t,i} = [ v_i ; f_snr(gamma) ] (one condition row per image)."""
        B       = ref_feats.shape[0]
        snr_emb = self.snr_encoder(snr_db, self.device).expand(B, -1)
        return torch.cat([ref_feats, snr_emb], dim=-1)

    # ---- UAP generation (Eq. (6) + norm clamp) -----------------------
    def generate_uap(self, ref_img_feats, snr_db):
        """Per-row text embedding perturbations, each row norm-clamped to
        [min_norm, max_norm]. Returns a (B, D) tensor."""
        cond    = self.build_cond(ref_img_feats, snr_db)
        uap_emb = self.generator(self.z, cond)
        uap_emb = uap_emb.reshape(uap_emb.size(0), -1)
        norm    = torch.norm(uap_emb, dim=1).view(-1, 1).expand_as(uap_emb)
        cn      = torch.clamp(norm, self.min_norm, self.max_norm)
        return cn * uap_emb / (norm + 1e-8)

    def tokenize(self, tokenizer, texts):
        return tokenizer(texts, padding='max_length', truncation=True,
                         max_length=self.max_length,
                         return_tensors='pt').to(self.device)

    @staticmethod
    def encode(model, text_inputs, uap_emb, cached_positions, cached_padding_zeros):
        out = model.inference_text_replace(text_inputs, uap_emb,
                                           cached_positions, cached_padding_zeros)
        return F.normalize(out['text_feat'].flatten(1), dim=-1)

    # ---- primary update -----------------------------------------------
    def attack(self, model, tokenizer, texts, txt2img,
               ref_img_feats, clean_img_feats, clean_txt_feats,
               cached_positions, cached_padding_zeros):
        """One text-branch update on the primary surrogate.

        Draws its own M stratified SNR values, averages the ACMC loss on the
        AWGN-corrupted adversarial text embeddings over the strata and takes one
        optimizer step. Strata whose loss has no gradient path to the
        perturbation (no substituted position in the batch) are skipped.

        Returns (loss, last_emb), where last_emb is the (B, D) row-wise
        perturbation of the last stratum.
        """
        device = self.device

        self.generator.train()
        self.snr_encoder.train()
        self.optimG.zero_grad()
        self.optimZ.zero_grad()

        snr_list = stratified_snr_sample(self.snr_min, self.snr_max,
                                         self.num_snr_strata)

        clean_v     = F.normalize(clean_img_feats[txt2img].to(device), dim=-1)
        clean_u     = F.normalize(clean_txt_feats[txt2img].to(device), dim=-1)
        text_inputs = self.tokenize(tokenizer, texts)

        losses   = []
        last_emb = None

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for snr_db in snr_list:
                uap_emb  = self.generate_uap(ref_img_feats, snr_db)
                last_emb = uap_emb

                adv_feats = self.encode(model, text_inputs,
                                        uap_emb[0:self.text_budget],
                                        cached_positions, cached_padding_zeros)

                noisy, _     = self.channel(adv_feats, snr_db)
                t_img, t_txt = select_farthest_target(noisy, clean_v, clean_u)
                loss = acmc_loss(noisy, t_img, t_txt, clean_v, clean_u,
                                 temperature=self.temperature, margin=self.margin)
                if loss.requires_grad:
                    losses.append(loss)

            if not losses:
                return torch.tensor(0.0, device=device), last_emb
            total_loss = torch.stack(losses).mean()

        if total_loss.requires_grad:
            total_loss.backward()
            self.optimG.step()
            self.optimZ.step()

        return total_loss.detach(), last_emb

    # ---- checkpointing ------------------------------------------------
    def save_model(self, path):
        torch.save({'generator':   self.generator.state_dict(),
                    'snr_encoder': self.snr_encoder.state_dict()}, path)

    def save_z(self, path):
        torch.save(self.z.data, path)

    def load_model(self, path, device='cuda'):
        d = torch.load(path, map_location=device)
        self.generator.load_state_dict(d['generator'])
        self.snr_encoder.load_state_dict(d['snr_encoder'])

    def load_z(self, path, device='cuda'):
        self.z.data = torch.load(path, map_location=device)
