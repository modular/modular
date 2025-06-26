# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test that symbolic dimension mismatches are reported when running a model."""

import numpy as np
import pytest
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


def test_symbolic_dimension_mismatch():
    device = CPU()

    data = Tensor.from_numpy(np.zeros((3, 4), dtype=np.float32))

    with Graph(
        "symbolic_dimension_mismatch",
        input_types=[
            TensorType(
                DType.float32,
                ("seq_len", "seq_len"),
                device=DeviceRef.from_device(device),
            ),
        ],
        outputs=[data],
    ) as graph:
        val = graph.inputs[0].tensor
        graph.output(val)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    # Passing a 3x4 tensor to a graph expecting 'seq_len'x'seq_len'.
    with pytest.raises(
        ValueError,
        match="symbolic dimension 'seq_len' for input 0 does not match prior uses of that dimension",
    ):
        model(data)
