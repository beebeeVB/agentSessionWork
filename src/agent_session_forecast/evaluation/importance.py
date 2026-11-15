"""Which features carry signal on accounts the model has never seen?"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .. import config as cfg
from ..features import features_at_k
from ..models import new_gbm


def grouped_importance(df, k=5, reps=3):
    """Permutation importance on held-out ACCOUNTS (MAE increase when shuffled)."""
    out = features_at_k(df, k)
    if out is None:
        return None
    X, y, groups = out["X"], out["y"], out["groups"]
    users = np.array(sorted(set(groups)))
    if len(users) < 10:
        return None
    imps = []
    for r in range(reps):
        perm = cfg.RNG.permutation(users)
        te = np.isin(groups, perm[: int(.3 * len(perm))])
        tr = ~te
        if te.sum() < 30 or tr.sum() < 50:
            continue
        m = new_gbm().fit(X[tr], y[tr])
        pi = permutation_importance(m, X[te], y[te], scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=r)
        imps.append(pi.importances_mean)
    if not imps:
        return None
    return pd.Series(np.mean(imps, 0), index=X.columns).sort_values(ascending=False)
