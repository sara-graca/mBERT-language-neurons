"""Compute and cache the activation probabilities P^k (12 x 3072) on dev.
"""

from datasets import load_dataset
from common import model, encode, cached, LANGS, FFNHooks


def load_flores(code, split):
    df = load_dataset("openlanguagedata/flores_plus", code, split=split).to_pandas()
    df["id"] = df["id"].astype(int)
    return df.sort_values("id").reset_index(drop=True)


def activation_probabilities(sentences, batch_size=32):
    with FFNHooks(model) as hooks:
        for i in range(0, len(sentences), batch_size):
            enc, keep = encode(sentences[i:i + batch_size])
            hooks.keep, hooks.n_tokens = keep, hooks.n_tokens + int(keep.sum())
            model.bert(**enc)
    return (hooks.counts / hooks.n_tokens).numpy()


if __name__ == "__main__":
    for code in LANGS:
        sents = load_flores(code, "dev")["text"].tolist()
        P = cached(f"P_{code}", lambda: activation_probabilities(sents))
        print(code, P.shape, f"mean p = {P.mean():.3f}")