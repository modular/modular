# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.engine Python bindings with MOF."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run

import max.engine as me
import numpy as np
import pytest

DYLIB_FILE_EXTENSION = "dylib" if os.uname().sysname == "Darwin" else "so"


# This path is used in skipif clauses rather than tests, so we can neither mark
# it as a fixture nor can we call other fixtures.
def modular_lib_path() -> Path:
    return Path(os.getenv("MODULAR_PATH")) / ".derived" / "build" / "lib"


@pytest.fixture
def mo_custom_ops_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "EngineAPI"
        / "c"
        / "custom-ops-override.api"
    )


@pytest.fixture
def sdk_test_inputs_path(modular_path: Path) -> Path:
    return modular_path / "SDK" / "integration-test" / "EngineAPI" / "Inputs"


@pytest.fixture
def relu_onnx_model_path(sdk_test_inputs_path: Path) -> Path:
    return sdk_test_inputs_path / "relu3x100x100.onnx"


@pytest.fixture
def relu_torchscript_model_path(sdk_test_inputs_path: Path) -> Path:
    return sdk_test_inputs_path / "relu3x100x100.torchscript"


@pytest.fixture
def custom_ops_package_path(request) -> Path:
    return Path(request.config.getoption("--custom-ops-path"))


@pytest.fixture
def mo_listio_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated model with list I/O."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "EngineAPI"
        / "Inputs"
        / "mo-list-model.mlir"
    )


def test_execute_success(mo_model_path: Path):
    session = me.InferenceSession()
    model = session.load(mo_model_path)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0, 2.0, -5.0, 3.0, 6.0]).astype(np.float32),
    )


@pytest.mark.skipif(
    not os.path.exists(modular_lib_path() / f"libmonnx.{DYLIB_FILE_EXTENSION}")
    or not os.path.exists(
        modular_lib_path() / f"libmtorch.{DYLIB_FILE_EXTENSION}"
    ),
    reason="One or more missing framework libs",
)
def test_execute_multi_framework(
    relu_onnx_model_path: Path,
    relu_torchscript_model_path: Path,
):
    session = me.InferenceSession()
    trch_input_specs = [
        me.TorchInputSpec(shape=[1, 3, 100, 100], dtype=me.DType.float32)
    ]
    onnx_model = session.load(relu_onnx_model_path)
    trch_model = session.load(
        relu_torchscript_model_path, input_specs=trch_input_specs
    )
    np_input = np.ones((1, 3, 100, 100))
    np_input[:, 1, :, :] *= -1
    onnx_output = onnx_model.execute(x=np_input)["result0"]
    trch_output = trch_model.execute(x=np_input)["result0"]
    assert np.allclose(onnx_output, trch_output)


def test_execute_gpu(mo_model_path: Path):
    output = run("is-cuda-available")
    if output.returncode != 0:
        return
    session = me.InferenceSession(device="cuda")
    model = session.load(mo_model_path)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0, 2.0, -5.0, 3.0, 6.0]).astype(np.float32),
    )


def test_custom_ops(
    mo_custom_ops_model_path: Path, custom_ops_package_path: Path
):
    session = me.InferenceSession()
    model = session.load(mo_custom_ops_model_path)
    inputs = np.ones((1)) * 4
    output = model.execute(input0=inputs)
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([2.0]).astype(np.float32),
    )

    model_with_custom_op = session.load(
        mo_custom_ops_model_path, custom_ops_path=str(custom_ops_package_path)
    )
    inputs = np.ones((1)) * 4
    output = model_with_custom_op.execute(input0=inputs)
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0]).astype(np.float32),
    )


def test_list_io(mo_listio_model_path: Path):
    session = me.InferenceSession()
    model_with_list_io = session.load(mo_listio_model_path)
    output = model_with_list_io.execute(
        input_list=[np.zeros(2)], input_tensor=np.ones(5)
    )
    assert "output_list" in output
    output_list = output["output_list"]
    assert len(output_list) == 3
    assert np.allclose(output_list[0], np.zeros(2))
    assert np.allclose(output_list[1], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(output_list[2], np.ones(5))


def test_dynamic_rank_spec():
    input_spec = me.TensorSpec(None, me.DType.float64, "dynamic")
    assert input_spec.shape is None
    assert input_spec.dtype == me.DType.float64
    assert input_spec.name == "dynamic"

    assert (
        repr(input_spec)
        == "TensorSpec(shape=None, dtype=DType.float64, name=dynamic)"
    )
    assert str(input_spec) == "None x float64"


def test_repr_torch_input_spec():
    input_spec_with_shape = me.TorchInputSpec([20, 30], me.DType.float32)
    assert input_spec_with_shape.shape == [20, 30]
    assert input_spec_with_shape.dtype == me.DType.float32

    assert (
        repr(input_spec_with_shape)
        == "TorchInputSpec(shape=[20, 30], dtype=DType.float32)"
    )
    assert str(input_spec_with_shape) == "20x30xfloat32"

    input_spec_without_shape = me.TorchInputSpec(None, me.DType.float64)
    assert input_spec_without_shape.shape == None
    assert input_spec_without_shape.dtype == me.DType.float64

    assert (
        repr(input_spec_without_shape)
        == "TorchInputSpec(shape=None, dtype=DType.float64)"
    )
    assert str(input_spec_without_shape) == "None x float64"

    input_spec_with_dim_names = me.TorchInputSpec(
        ["BATCH", 30], me.DType.float32
    )
    assert input_spec_with_dim_names.shape == ["BATCH", 30]
    assert (
        repr(input_spec_with_dim_names)
        == "TorchInputSpec(shape=['BATCH', 30], dtype=DType.float32)"
    )
    assert str(input_spec_with_dim_names) == "-1x30xfloat32"
