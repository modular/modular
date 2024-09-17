# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test value printing and debug print options."""


import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.graph import Graph, TensorType


@pytest.fixture(scope="module")
def compiled_model(session):
    def print_input(x):
        x.print("test_x_value")
        return x

    g = Graph("test_print", print_input, [TensorType(DType.float32, ["dim1"])])
    return session.load(g)


def test_debug_print_options(session, tmp_path):
    with pytest.raises(TypeError, match="Invalid debug print style"):
        session.set_debug_print_options("NOTVALID")

    session.set_debug_print_options("COMPACT")
    session.set_debug_print_options("NONE")

    session.set_debug_print_options("FULL")
    session.set_debug_print_options("FULL", precision=5)
    with pytest.raises(TypeError, match="precision must be an int"):
        session.set_debug_print_options("FULL", precision=5.0)

    session.set_debug_print_options("BINARY", output_directory=tmp_path)
    session.set_debug_print_options("BINARY", output_directory=str(tmp_path))
    with pytest.raises(ValueError, match="output directory cannot be empty"):
        session.set_debug_print_options("BINARY")


def test_debug_print_compact(compiled_model, session, capfd):
    session.set_debug_print_options("COMPACT")
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32))
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value_cpu = tensor([[1.1235, 1.1235, 1.1235, ..., 1.1235,"
        " 1.1235, 1.1235]], dtype=f32, shape=[20])"
        in captured.out
    )


def test_debug_print_full(compiled_model, session, capfd):
    session.set_debug_print_options("FULL", 2)
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32))
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value_cpu = tensor<20xf32> [1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00, 1.12e+00,"
        " 1.12e+00, 1.12e+00, 1.12e+00]"
        in captured.out
    )

    session.set_debug_print_options("FULL", 6)
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32))
    )
    captured = capfd.readouterr()
    assert (
        "test_x_value_cpu = tensor<20xf32> [1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00, 1.123457e+00,"
        " 1.123457e+00, 1.123457e+00, 1.123457e+00]"
        in captured.out
    )


def test_debug_print_none(compiled_model, session, capfd):
    session.set_debug_print_options("NONE")
    _ = compiled_model.execute(
        Tensor.from_numpy(np.full([20], 1.1234567, np.float32))
    )
    captured = capfd.readouterr()
    assert "test_x_value_cpu" not in captured.out


def test_debug_print_binary(compiled_model, session, capfd, tmp_path):
    session.set_debug_print_options("BINARY", output_directory=tmp_path)
    input = np.full([20], 1.1234567, np.float32)
    _ = compiled_model.execute(Tensor.from_numpy(input))
    captured = capfd.readouterr()
    assert "test_x_value_cpu" not in captured.out
    assert (tmp_path / "test_x_value_cpu").exists()
    from_file = np.fromfile(tmp_path / "test_x_value_cpu", np.float32)
    assert (input == from_file).all()
