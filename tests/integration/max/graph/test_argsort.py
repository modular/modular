# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import cast

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize(
    ("input_shape", "ascending"),
    [
        ((5,), True),
        ((4,), True),
        ((2,), True),
        ((6,), False),
        ((3,), False),
    ],
)
def test_argsort_execution(
    session: InferenceSession,
    input_shape: tuple[int, ...],
    ascending: bool,
) -> None:
    """Tests end-to-end argsort lowering and execution against torch.argsort."""
    graph = Graph(
        "argsort_test",
        forward=lambda x: ops.argsort(x, ascending=ascending),
        input_types=[
            TensorType(
                DType.float32,
                shape=input_shape,
                device=DeviceRef.from_device(session.devices[0]),
            )
        ],
    )

    # Compile and execute the graph.
    model = session.load(graph)

    # Generate random input data.
    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input)

    # Execute MAX model.
    max_indices = model.execute(
        Tensor.from_numpy(np_input).to(model.input_devices[0])
    )[0]

    # Get torch reference results.
    torch_indices = torch.argsort(
        torch_input,
        descending=not ascending,
    )

    # For indices verification, check that gathered values match.
    # This handles potential index ordering differences when sorted=False
    # (once this test supports that).
    np.testing.assert_array_equal(
        cast(Tensor, max_indices).to_numpy(),
        torch_indices.numpy(),
    )
