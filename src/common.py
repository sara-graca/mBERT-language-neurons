"""Shared setup for Part 3: model, tokeniser, cache helper, encode, FFNHooks."""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_ID = "google-bert/bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"

tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION).eval()
torch.set_grad_enabled(False)

LANGS = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hans", "yor_Latn"]
os.makedirs("cache", exist_ok=True)


def cached(name, fn):
    """Load cache/<name>.npy if it exists; otherwise compute fn(), save, return."""
    path = f"cache/{name}.npy"
    if not os.path.exists(path):
        np.save(path, fn())
    return np.load(path)

def encode(sentences, max_length=256):
    enc = tok(sentences, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt", return_special_tokens_mask=True)
    special = enc.pop("special_tokens_mask").bool()
    keep = enc["attention_mask"].bool() & ~special
    return enc, keep


class FFNHooks:
    """Counts positive GELU activations and/or zeroes selected FFN neurons."""
    def __init__(self, model, ablate=None):
        self.layers = model.bert.encoder.layer
        self.counts = torch.zeros(len(self.layers), model.config.intermediate_size)
        self.n_tokens = 0
        self.keep = None
        self.ablate = ablate or {}

    def _hook(self, i):
        def hook(module, inputs, output):
            if self.keep is not None:
                active = (output > 0) & self.keep.unsqueeze(-1)
                self.counts[i] += active.sum(dim=(0, 1))
            idx = self.ablate.get(i)
            if idx is not None and len(idx) > 0:
                output = output.clone()
                output[..., idx] = 0.0
            return output
        return hook

    def __enter__(self):
        self.handles = [layer.intermediate.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()