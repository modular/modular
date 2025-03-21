# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings for mojo analysis."""

from pathlib import Path

import pytest
from max import mlir
from max.graph import Graph


def test_kernel_library(counter_mojopkg):
    with Graph("test_kernel_library") as graph:
        kernels = graph._kernel_library
        kernels.add_path(counter_mojopkg)

        assert "make_counter" in kernels
        assert isinstance(kernels["make_counter"], mlir.Operation)

        with pytest.raises(ValueError) as err:
            kernels.add_path(Path("/path/to/invalid.mojopkg"))
        assert "No such file or directory" in str(err.value)
