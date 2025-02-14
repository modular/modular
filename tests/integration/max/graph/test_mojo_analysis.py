# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings for mojo analysis."""

from max import mlir
from max.graph import Graph


def test_mojo_analysis(counter_mojopkg):
    with Graph("test_mojo_analysis") as graph:
        kernels = graph.import_kernels(counter_mojopkg)
        assert "make_counter" in kernels
        assert isinstance(kernels["make_counter"], mlir.Operation)
