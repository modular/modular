# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from conftest import tensor_types
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType

shared_types = st.shared(tensor_types())


def test_transfer_to_basic() -> None:
    target_device = DeviceRef.GPU()
    with Graph(
        "transfer",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.CPU()
            ),
        ],
    ) as graph:
        out = graph.inputs[0].to(target_device)
        assert out.device == target_device
        graph.output(out)


def test_transfer_identity() -> None:
    with Graph(
        "identity",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.GPU()
            ),
        ],
    ) as graph:
        # gpu:0 --> gpu:0 is useless so this should be no-op
        out = graph.inputs[0].to(DeviceRef.GPU())
        graph.output(out)

    # make sure no transfer operation was created
    assert "transfer" not in str(graph)
