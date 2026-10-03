"""Concept directions and decompositions with held-out evaluation contracts."""
import numpy as np
import torch
from .representations import _matrix


def fit_cav(positive, negative, *, regularization=1.0, seed=7):
    """Linear concept separator trained only on caller-provided reference data.

    Evaluate on disjoint examples, repeat with random concepts and label
    permutations before claiming a TCAV finding. Direction is in original units.
    """
    from sklearn.linear_model import LogisticRegression
    pos, neg = _matrix(positive).cpu().numpy(), _matrix(negative).cpu().numpy()
    if pos.shape[1] != neg.shape[1] or regularization <= 0:
        raise ValueError("Equal feature widths and positive regularization required")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    classifier = LogisticRegression(C=regularization, random_state=seed, max_iter=500).fit(x,y)
    direction = torch.from_numpy(classifier.coef_[0].copy()).float()
    if direction.norm() <= 1e-12:
        raise ValueError("Degenerate concept direction")
    return {"direction": direction/direction.norm(), "classifier": classifier,
            "training_accuracy": float(classifier.score(x,y))}


def nmf_features(activations, *, components=3, seed=7):
    """NMF of nonnegative [observations,features]; learned axes are not semantic labels."""
    from sklearn.decomposition import NMF
    x = _matrix(activations).cpu().numpy()
    if (x < 0).any() or not 1 <= components <= min(x.shape):
        raise ValueError("NMF requires nonnegative data and bounded component count")
    estimator = NMF(n_components=components, init='nndsvda', random_state=seed, max_iter=300)
    codes = estimator.fit_transform(x)
    return {"codes": codes, "components": estimator.components_,
            "reconstruction_error": float(estimator.reconstruction_err_)}
