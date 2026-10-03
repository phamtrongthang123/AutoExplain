import pytest
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.tree import DecisionTreeClassifier
from autoexplain.tabular import explain_tree_path


def test_path_matches_sklearn_leaf_and_prediction():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    model = DecisionTreeClassifier(random_state=4).fit(x, [0, 0, 1, 1])
    for row in x:
        result = explain_tree_path(model, row, ["a", "b"])
        assert result["leaf"] == model.apply(row[None])[0]
        assert result["prediction"] == model.predict(row[None])[0]
        assert sum(result["probabilities"]) == pytest.approx(1)
        for step in result["path"]:
            assert (step["value"] <= step["threshold"]) == (step["operator"] == "<=")


def test_invalid_rows_and_names():
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    for row in [[1, 2], [float("nan")], [[1]]]:
        with pytest.raises(ValueError):
            explain_tree_path(model, row)
    with pytest.raises(ValueError):
        explain_tree_path(model, [0], [])
