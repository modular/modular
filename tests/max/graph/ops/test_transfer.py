# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from conftest import tensor_types
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Device, Graph, TensorType


shared_types = st.shared(tensor_types())


def test_transfer_to_basic() -> None:
    target_device = Device.CUDA()
    with Graph(
        "transfer",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=Device.CPU()),
        ],
    ) as graph:
        out = graph.inputs[0].to(target_device)
        assert out.device == target_device
        graph.output(out)
