"""Sequence models on the request stream (``--sequence``).

Two ways of letting a model see the *order* of the first k requests, not just their sums:

* ``seq_gru``    - a small GRU over the per-request feature sequence (tokens, cache, latency,
                   tool calls/errors, gap between requests, and with ``--feature-set rich`` also
                   trigger/mutation/tool result size/latency) concatenated with the static
                   features, then a small MLP head. Needs PyTorch; if PyTorch is missing (or the
                   GRU fails) it falls back to ``sklearn``'s MLP on the flattened sequence and the
                   predictor is then named ``seq_mlp`` in the output.
* ``tool_ngram_gbm`` - gradient boosting on the usual features plus counts of the most common
                   per-request tool-signature tokens and their bigrams (a bag-of-n-grams).

Hyper-parameters are fixed and modest, everything is seeded; nothing is tuned to win. At a few
thousand sessions a tie with the plain GBM is the *expected* outcome, and a valid result.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

from .. import config as cfg
from ..models import new_gbm, prep_linear
from .compare import evaluate_predictors, fmt_paired


def seq_tensor(d, k):
    """(n_sessions, k, channels) array from the first k requests only."""
    A = lambda col: np.array([a[:k] for a in d[col]], dtype=float)
    st = A("a_start")
    gap = np.diff(st, axis=1, prepend=st[:, :1]) / 1000.0
    ch = [np.log1p(A("a_completion")), np.log1p(A("a_prompt")), np.log1p(A("a_cached")),
          np.log1p(np.maximum(A("a_ttft"), 0)), A("a_ntools"), A("a_nerr"), np.log1p(np.maximum(gap, 0))]
    if cfg.FEATURE_SET == "rich" and "a_user" in d.columns:
        ch += [A("a_user"), A("a_mut"), np.log1p(A("a_restok")), np.log1p(A("a_lat"))]
    return np.stack(ch, axis=-1)


def _standardise(S_tr, S_te):
    mu, sd = S_tr.reshape(-1, S_tr.shape[-1]).mean(0), S_tr.reshape(-1, S_tr.shape[-1]).std(0) + 1e-6
    return (S_tr - mu) / sd, (S_te - mu) / sd


def _fit_predict_gru(S_tr, Z_tr, y_tr, S_te, Z_te, epochs):
    import torch
    from torch import nn
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    class Net(nn.Module):
        def __init__(self, c, z):
            super().__init__()
            self.gru = nn.GRU(c, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + z, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, s, z):
            _, h = self.gru(s)
            return self.head(torch.cat([h[-1], z], 1)).squeeze(1)

    f = lambda a: torch.tensor(a, dtype=torch.float32)
    St, Zt, yt = f(S_tr), f(Z_tr), f(y_tr - y_tr.mean())
    net = Net(S_tr.shape[-1], Z_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        for i in range(0, len(yt), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            nn.functional.l1_loss(net(St[b], Zt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(f(S_te), f(Z_te)).numpy() + y_tr.mean()


_USED = {"name": "seq_gru"}


def pred_sequence(out, tr, te, ctx):
    X, y, d, k = out["X"], out["y"], out["d"], ctx["k"]
    S = seq_tensor(d, k)
    S_tr, S_te = _standardise(S[tr], S[te])
    Z_tr, Z_te = prep_linear(X.iloc[tr], X.iloc[te])
    try:
        p = _fit_predict_gru(S_tr, Z_tr, y[tr], S_te, Z_te, cfg.SEQ_EPOCHS)
        _USED["name"] = "seq_gru"
        return p
    except Exception as e:
        if not isinstance(e, ImportError):
            print(f"    [note] GRU failed ({type(e).__name__}: {e}); using MLP fallback")
        from sklearn.neural_network import MLPRegressor
        A = np.hstack([S_tr.reshape(len(tr), -1), Z_tr])
        B = np.hstack([S_te.reshape(len(te), -1), Z_te])
        m = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=200, random_state=0)
        _USED["name"] = "seq_mlp"
        return m.fit(A, y[tr]).predict(B)


def _tokens(d, k):
    """Per-request signature tokens (the sorted tool-category string) for the first k requests."""
    return [[c if c else "<none>" for c in a[:k]] for a in d["a_cats"]]


def pred_ngram(out, tr, te, ctx, top_uni=25, top_bi=25):
    X, y, d, k = out["X"], out["y"], out["d"], ctx["k"]
    toks = _tokens(d, k)
    uni = Counter(t for i in tr for t in toks[i])
    bi = Counter(b for i in tr for b in zip(toks[i][:-1], toks[i][1:]))
    U = [t for t, _ in uni.most_common(top_uni)]
    B = [b for b, _ in bi.most_common(top_bi)]

    def feats(idx):
        M = np.zeros((len(idx), len(U) + len(B)))
        for r, i in enumerate(idx):
            cu, cb = Counter(toks[i]), Counter(zip(toks[i][:-1], toks[i][1:]))
            for c, t in enumerate(U):
                M[r, c] = cu.get(t, 0)
            for c, b in enumerate(B):
                M[r, len(U) + c] = cb.get(b, 0)
        return pd.DataFrame(M, columns=[f"ng{c}" for c in range(M.shape[1])])

    Xtr = pd.concat([X.iloc[tr].reset_index(drop=True), feats(tr)], axis=1)
    Xte = pd.concat([X.iloc[te].reset_index(drop=True), feats(te)], axis=1)
    return new_gbm().fit(Xtr, y[tr]).predict(Xte)


def run(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    res, paired, store = evaluate_predictors(df, {"sequence": pred_sequence, "tool_ngram_gbm": pred_ngram})
    used = _USED["name"]
    for t in (res, paired):
        for c in ("model",):
            if c in t:
                t[c] = t[c].replace({"sequence": used})
    tables = {"seq_forecast": res, "seq_paired": paired}
    for n, t in tables.items():
        if len(t):
            t.to_csv(os.path.join(outdir, n + ".csv"), index=False)
    L = ["", f"SEQUENCE MODELS (neural sequence model = {used}; fixed hyper-parameters, not tuned):"]
    L += fmt_paired(paired, designs=("grouped", "random"), refs=("gbm",))
    return dict(tables=tables, lines=L, payload=dict(res=res, paired=paired, used=used))
