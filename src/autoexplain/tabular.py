"""Decision-tree path explanations with a lazy scikit-learn dependency."""


def explain_tree_path(model, row, feature_names=None):
    """Describe one sklearn DecisionTreeClassifier decision path.

    Returns actual branch predicates and leaf class probabilities, not feature
    attribution or SHAP values. Only one finite numeric row is supported.
    """
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError("Expected sklearn.tree.DecisionTreeClassifier")
    check_is_fitted(model)
    if model.n_outputs_ != 1:
        raise ValueError("Only single-output classifiers are supported")
    values = np.asarray(row, dtype=np.float32)
    if values.shape != (model.n_features_in_,) or not np.isfinite(values).all():
        raise ValueError("row must be one finite vector matching model features")
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(len(values))]
    if len(names) != len(values):
        raise ValueError("feature_names must match feature count")
    tree = model.tree_
    node = 0
    path = []
    while tree.children_left[node] != tree.children_right[node]:
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        left = bool(values[feature] <= threshold)
        path.append({"node": node, "feature": names[feature], "feature_index": feature,
                     "value": float(values[feature]), "operator": "<=" if left else ">",
                     "threshold": threshold})
        node = int(tree.children_left[node] if left else tree.children_right[node])
    return {"path": path, "leaf": node, "prediction": model.predict(values[None])[0].item(),
            "classes": model.classes_.tolist(),
            "probabilities": model.predict_proba(values[None])[0].tolist()}
