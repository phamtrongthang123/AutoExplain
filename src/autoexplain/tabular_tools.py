"""Local tabular diagnostics with explicit data and bounded searches."""
import numpy as np


def permutation_importance(model, x, y, *, repeats=5, scoring=None, seed=7):
    from sklearn.inspection import permutation_importance as compute
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be 1–100")
    return compute(model, x, y, n_repeats=repeats, scoring=scoring, random_state=seed, n_jobs=1)


def partial_dependence_ice(model, x, features, *, grid_resolution=10):
    """Brute PDP + ICE; correlated-feature interventions can be off-distribution."""
    from sklearn.inspection import partial_dependence
    if not 2 <= grid_resolution <= 100:
        raise ValueError("grid_resolution must be 2–100")
    return partial_dependence(model, x, features, kind="both", method="brute",
                              grid_resolution=grid_resolution)


def nearest_counterfactual(model, row, candidates, *, target, mutable=None, scale=None):
    """Find nearest supplied feasible candidate classified as target.

    Not a generative optimizer or causal recourse guarantee. The caller must
    supply domain-valid candidates and constraints; immutable columns must match.
    """
    row, candidates = np.asarray(row, dtype=float), np.asarray(candidates, dtype=float)
    if row.ndim != 1 or candidates.ndim != 2 or candidates.shape[1] != row.size:
        raise ValueError("Expected row [D] and candidates [N,D]")
    if not np.isfinite(row).all() or not np.isfinite(candidates).all():
        raise ValueError("Inputs must be finite")
    mutable = np.ones(row.size, dtype=bool) if mutable is None else np.asarray(mutable, dtype=bool)
    scale = np.ones(row.size) if scale is None else np.asarray(scale, dtype=float)
    if mutable.shape != row.shape or scale.shape != row.shape or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid mutable mask or positive scale")
    valid = np.all(candidates[:, ~mutable] == row[~mutable], axis=1)
    filtered = candidates[valid]
    if not len(filtered):
        return None
    filtered = filtered[np.asarray(model.predict(filtered)) == target]
    if not len(filtered):
        return None
    distances = np.abs((filtered-row)/scale).sum(axis=1)
    index = int(distances.argmin())
    return {"counterfactual": filtered[index].copy(), "scaled_l1": float(distances[index]),
            "target": target, "feasible_candidates": len(filtered)}
