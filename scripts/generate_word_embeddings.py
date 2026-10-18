"""
Precompute the word-embedding / vocabulary caches read by train_semcom_uap.py
and eval_semcom_uap.py (the files read by the C-PGC code).

- word_embedding_{ALBEF,TCL,XVLM,BLIP}.pth: the fine-tuned ALBEF text-encoder
  word-embedding table (taken from ./checkpoint/ALBEF/flickr30k.pth). Training
  uses its min/max L2 norm to clamp the text perturbation; evaluation uses it
  for the nearest-word lookup that turns the text UAP into an adversarial word.
- word_embedding_CLIP.pth: the CLIP ViT-B/16 token-embedding table.
- word_idx_{TCL,XVLM,BLIP}.npy: aliases of the shipped word_idx_ALBEF.npy
  (candidate vocabulary for the nearest-word lookup). word_idx_CLIP.npy is
  shipped as well.

Usage (run once from the repo root, after downloading the ALBEF checkpoint):
    python scripts/generate_word_embeddings.py --all
"""
import argparse
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

BERT_ALIASES = ["TCL", "XVLM", "BLIP"]


def save_bert_word_embedding(albef_ckpt: str, out_path: str):
    checkpoint = torch.load(albef_ckpt, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    word_embedding = state_dict["text_encoder.embeddings.word_embeddings.weight"].detach().clone()
    torch.save(word_embedding, out_path)
    print(f"[bert] saved {tuple(word_embedding.shape)} -> {out_path}")


def save_clip_word_embedding(clip_model_name: str, out_path: str):
    from models import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(clip_model_name, device=device)
    word_embedding = model.token_embedding.weight.detach().clone().cpu()
    torch.save(word_embedding, out_path)
    print(f"[clip] saved {tuple(word_embedding.shape)} -> {out_path}")


def link_or_copy(src_name: str, dst_name: str):
    src = os.path.join(REPO_ROOT, src_name)
    dst = os.path.join(REPO_ROOT, dst_name)
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src_name, dst)
    except OSError:
        shutil.copyfile(src, dst)
    print(f"[alias] {dst_name} -> {src_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--albef_ckpt", default=os.path.join(REPO_ROOT, "checkpoint", "ALBEF", "flickr30k.pth"),
                        help="Fine-tuned ALBEF Flickr30K checkpoint whose text-encoder word embeddings "
                             "are shared by ALBEF/TCL/BLIP/XVLM")
    parser.add_argument("--clip_model", default="ViT-B/16",
                        help="CLIP variant whose token embedding table to cache")
    parser.add_argument("--all", action="store_true",
                        help="Generate word_embedding_{ALBEF,TCL,XVLM,BLIP,CLIP}.pth and word_idx aliases")
    args = parser.parse_args()

    if not args.all:
        raise SystemExit("pass --all (all BERT-based backbones share the ALBEF word-embedding table)")

    save_bert_word_embedding(args.albef_ckpt, os.path.join(REPO_ROOT, "word_embedding_ALBEF.pth"))
    for name in BERT_ALIASES:
        link_or_copy("word_embedding_ALBEF.pth", f"word_embedding_{name}.pth")
        link_or_copy("word_idx_ALBEF.npy", f"word_idx_{name}.npy")
    save_clip_word_embedding(args.clip_model, os.path.join(REPO_ROOT, "word_embedding_CLIP.pth"))
