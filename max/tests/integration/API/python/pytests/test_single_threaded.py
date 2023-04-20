# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the modular.engine on single-threaded scenarios"""

import pytest
from test_utils import compile_and_execute_on_random_input


@pytest.mark.parametrize(
    "model_path_fixture",
    [
        "tf_basic_mlp",
        "tf_efficientnet",
        "tf_dlrm",
        "tf_bert",
    ],
)
def test_single_threaded_execute_tensorflow(
    model_path_fixture: str, request
) -> None:
    """Test model execute has no errors."""
    # using fixtures this way enables lazy (dynamic) initialization with mark.parametrize
    # https://stackoverflow.com/questions/42014484/pytest-using-fixtures-as-arguments-in-parametrize
    compile_and_execute_on_random_input(
        request.getfixturevalue(model_path_fixture)
    )


@pytest.mark.parametrize(
    "model_path_fixture",
    [
        "onnx_dlrm",
        "onnx_bert",
    ],
)
def test_single_threaded_execute_onnx(model_path_fixture, request) -> None:
    """Test model execute has no errors."""
    compile_and_execute_on_random_input(
        request.getfixturevalue(model_path_fixture)
    )


@pytest.mark.parametrize(
    "model_path_fixture",
    ["tflite_basic_mlp", "tflite_camembert"],
)
def test_single_threaded_execute_tflite(model_path_fixture, request) -> None:
    """Test model execute has no errors."""
    compile_and_execute_on_random_input(
        request.getfixturevalue(model_path_fixture)
    )
