"""LAPE selection + ablation

    python ablate.py   ->  cache/sel.npy, baseline.npy, loss_lape.npy, loss_rand.npy

Neuron sets are boolean masks of shape (12, 3072): True = selected neuron.
"""

import numpy as np
import torch
import torch.nn.functional as F
from common import model, tok, encode, LANGS, FFNHooks
from compute import load_flores


def masked_batches(sentences, p=0.15, seed=0, batch_size=32):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode(sentences[i:i + batch_size])
        m = (torch.rand(keep.shape, generator=g) < p) & keep
        labels = enc["input_ids"].masked_fill(~m, -100)
        enc["input_ids"] = enc["input_ids"].masked_fill(m, tok.mask_token_id)
        batches.append((enc, labels))
    return batches


def mlm_loss(batches, ablate=None):
    total, n = 0.0, 0
    with FFNHooks(model, ablate=ablate):
        for enc, labels in batches:
            h = model.bert(**enc).last_hidden_state
            m = labels != -100
            logits = model.cls(h[m])
            total += F.cross_entropy(logits, labels[m], reduction="sum").item()
            n += int(m.sum())
    return total / n


def lape_select(P, top=0.01):
    """P: (L, 12, 3072) activation probabilities.
    Returns sel (L, 12, 3072) bool: neuron assigned to language k; lape; tau."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = P / P.sum(0)                            # normalise over languages
        lape = -np.nansum(q * np.log(q), axis=0)    # entropy; 0*log0 counts as 0
    lape[P.sum(0) == 0] = np.inf                    # never active: excluded
    kept = np.zeros(lape.shape, bool)
    kept.flat[np.argsort(lape, axis=None)[:int(top * lape.size)]] = True
    tau = np.percentile(P, 95)
    return kept & (P > tau), lape, tau


def random_like(mask, free, rng):
    """Same number of neurons per layer as `mask`, drawn among `free` neurons."""
    out = np.zeros_like(mask)
    for l in range(mask.shape[0]):
        out[l, rng.choice(np.flatnonzero(free[l]), mask[l].sum(), replace=False)] = True
    return out


def to_ablate(mask):
    """Boolean (12, 3072) mask -> {layer: LongTensor of neuron indices}."""
    return {l: torch.from_numpy(np.flatnonzero(m)) for l, m in enumerate(mask) if m.any()}


if __name__ == "__main__":
    P = np.stack([np.load(f"cache/P_{c}.npy") for c in LANGS]) 
    sel, lape, tau = lape_select(P)
    np.save("cache/sel.npy", sel)
    print("tau =", tau, "| neurons per language:", dict(zip(LANGS, sel.sum((1, 2)))))

    batches = {c: masked_batches(load_flores(c, "devtest")["text"].tolist()[:300])
               for c in LANGS}

    def losses(mask):  # loss on every language with `mask` zeroed
        return [mlm_loss(batches[c], to_ablate(mask)) for c in LANGS]

    np.save("cache/baseline.npy", losses(np.zeros_like(sel[0])))
    np.save("cache/loss_lape.npy", np.array([losses(m) for m in sel]))

    rng, free = np.random.default_rng(0), ~sel.any(0)
    np.save("cache/loss_rand.npy", np.array(
        [np.mean([losses(random_like(m, free, rng)) for _ in range(3)], axis=0)
         for m in sel]))
    print("done")