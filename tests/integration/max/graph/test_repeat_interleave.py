# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, Graph, StaticDim, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize(
    "input,repeats", [([1, 2, 3], 2), ([[1, 2], [3, 4]], 3)]
)
def test_repeat_interleave(
    session,
    input: TensorType,
    repeats: int,
):
    with Graph(
        "repeat_interleave",
        input_types=[],
    ) as graph:
        x = ops.constant(np.array(input), DType.int64).to(device_ref)

        output = ops.repeat_interleave(x, repeats)
        graph.output(output)

    expected = (
        torch.repeat_interleave(torch.tensor(input), repeats).detach().numpy()
    )

    model = session.load(graph)
    result = model.execute()[0]

    np.testing.assert_equal(result.to_numpy(), expected)


@pytest.mark.parametrize(
    "input,repeats,axis",
    [
        # 1-d with matching length repeats
        ([1, 2, 3], [2, 3, 4], 0),
        # 1-d with broadcasted repeats
        ([1, 2, 3, 4], [4], 0),
        # 2-d along either axis
        ([[1, 2], [3, 4]], [1, 2], 0),
        ([[1, 2], [3, 4]], [1, 2], 1),
    ],
)
def test_repeat_interleave_vector(
    session,
    input: TensorType,
    repeats: list[int],
    axis: int,
):
    with Graph(
        "repeat_interleave_vector",
        input_types=[],
    ) as graph:
        x = ops.constant(np.array(input), DType.int64).to(device_ref)
        repeat_vals = ops.constant(np.array(repeats), DType.int64).to(
            device_ref
        )

        if len(repeats) == 1:
            out_dim = x.shape[axis] * sum(repeats)
        else:
            out_dim = StaticDim(sum(repeats))

        output = ops.repeat_interleave(x, repeat_vals, axis, out_dim=out_dim)
        graph.output(output)

    expected = (
        torch.repeat_interleave(
            torch.tensor(input), torch.tensor(repeats), dim=axis
        )
        .detach()
        .numpy()
    )

    model = session.load(graph)
    result = model.execute()[0]

    np.testing.assert_equal(result.to_numpy(), expected)
