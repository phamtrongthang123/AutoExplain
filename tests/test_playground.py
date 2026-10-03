"""Optional Streamlit runtime smoke coverage, no server or shared model cache."""
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_playground_methods_and_controls():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "playground/app.py"), default_timeout=60)
    app.run()
    assert not app.exception
    for method in ["Integrated gradients", "SmoothGrad", "Occlusion", "LayerCAM"]:
        app.selectbox[1].set_value(method).run()
        assert not app.exception
    app.checkbox[0].check().run()
    assert not app.exception
    app.checkbox[1].check().run()
    assert not app.exception


def test_optional_cam_playground():
    pytest.importorskip('pytorch_grad_cam')
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'playground/app.py'), default_timeout=60).run()
    app.checkbox[2].check().run()
    assert not app.exception
    app.selectbox[2].set_value('hirescam').run()
    assert not app.exception
