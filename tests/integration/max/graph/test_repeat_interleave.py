# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
import torch
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, StaticDim, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.skipif(
    accelerator_count() > 0, reason="repeat_interleave is not supported on GPU"
)
@pytest.mark.parametrize(
    "input,repeats", [([1, 2, 3], 2), ([[1, 2], [3, 4]], 3)]
)
def test_repeat_interleave(
    session: InferenceSession,
    input: TensorType,
    repeats: int,
) -> None:
    with Graph(
        "repeat_interleave",
        input_types=[],
    ) as graph:
        x = ops.constant(np.array(input), DType.int64, device=device_ref)

        output = ops.repeat_interleave(x, repeats)
        graph.output(output)

    expected = (
        torch.repeat_interleave(torch.tensor(input), repeats).detach().numpy()
    )

    model = session.load(graph)
    result = model.execute()[0]
    assert isinstance(result, Tensor)

    np.testing.assert_equal(result.to_numpy(), expected)


@pytest.mark.skipif(
    accelerator_count() > 0, reason="repeat_interleave is not supported on GPU"
)
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
    session: InferenceSession, input: TensorType, repeats: list[int], axis: int
) -> None:
    with Graph(
        "repeat_interleave_vector",
        input_types=[],
    ) as graph:
        x = ops.constant(np.array(input), DType.int64, device_ref)
        repeat_vals = ops.constant(
            np.array(repeats), DType.int64, DeviceRef.CPU()
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
    assert isinstance(result, Tensor)

    np.testing.assert_equal(result.to_numpy(), expected)
