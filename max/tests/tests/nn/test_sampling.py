# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.sampling"""

from typing import cast

import numpy as np
import pytest
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.sampling import MinPSampler


# NOTE THAT ONLY RANK 2 TENSORS
# ARE CURRENTLY SUPPORTED
@pytest.mark.parametrize(
    ("input_shape", "min_p", "temperature"),
    [
        ((5, 5), 0.0, 0.0),
        ((4, 3), 0.01, 0.1),
        ((2, 3), 0.05, 0.2),
        ((6, 2), 0.08, 0.25),
        ((3, 5), 0.1, 0.3),
    ],
)
def test_min_p_execution(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    min_p: float,
    temperature: float,
) -> None:
    """Tests end-to-end MinPSampling lowering and execution."""
    with Graph(
        "min_p_test",
        input_types=[
            TensorType(DType.float32, shape=input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        inputs, *_ = graph.inputs
        sampler = MinPSampler(DType.float32, input_shape, min_p, temperature)
        out = sampler(inputs.tensor)
        graph.output(out)

        # Compile and execute the graph.
        model = session.load(graph)

        # Generate random input data.
        np_input = np.random.randn(*input_shape).astype(np.float32)

        # Execute MAX model.
        min_p_output, *_ = model.execute(np_input)
        cast(Tensor, min_p_output).to_numpy()
