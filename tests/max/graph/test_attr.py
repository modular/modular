# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""
import array
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from max import _graph
from max.graph import DType, TensorType


def test_array_attr(mlir_context) -> None:
    """Tests array attribute creation."""
    buffer = array.array("f", [42, 3.14])

    array_attr = _graph.array_attr(
        "foo", buffer, TensorType(DType.float32, (2,)).to_mlir()
    )
    assert "dense_array" in str(array_attr)


def test_weights_attr(mlir_context) -> None:
    """Tests weighst attributes creation."""
    with NamedTemporaryFile("wb") as weights_file:
        weights_file.write(b"Hello, world!\n")

        weights_attr = _graph.weights_attr(
            Path(weights_file.name),
            0,
            TensorType(DType.uint8, (2, 2)).to_mlir(),
            "bar",
        )
        assert "dense_resource" in str(weights_attr)


def test_weights_attr_invalid_path(mlir_context) -> None:
    """Tests that an error is thrown if a path doesn't exist."""
    with NamedTemporaryFile("wb") as weights_file:
        # Close the file, which removes it.
        weights_file.close()

        with pytest.raises(RuntimeError, match="No such file or directory"):
            _graph.weights_attr(
                Path(weights_file.name),
                0,
                TensorType(DType.uint8, ((1,))).to_mlir(),
                "bar",
            )


def test_dim_param_decl_attr(mlir_context) -> None:
    """Tests dim param declaration attribute creation."""
    attr = _graph.dim_param_decl_attr(mlir_context, "dim1")
    assert "param.decl dim1" in str(attr)


def test_dim_param_decl_array_attr(mlir_context) -> None:
    """Tests dim param declaration array attribute creation."""
    dim1 = _graph.dim_param_decl_attr(mlir_context, "dim1")
    dim2 = _graph.dim_param_decl_attr(mlir_context, "dim2")
    attr = _graph.dim_param_decl_array_attr(mlir_context, [dim1, dim2])
    assert re.search(r"param\.decls.*dim1.*dim2", str(attr))


def test_shape_attr(mlir_context) -> None:
    """Tests shape attribute creation."""
    dim1 = _graph.static_dim(mlir_context, 3)
    dim2 = _graph.symbolic_dim(mlir_context, "x")
    attr = _graph.shape_attr(mlir_context, [dim1, dim2])
    assert "mosh<ape[3, x]" in str(attr)
