# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, StaticDim, TensorType, ops


@pytest.mark.parametrize("device_ref", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize("dtype", [DType.int64, DType.float32])
@pytest.mark.parametrize(
    "input,repeats",
    [([1, 2, 3], 0), ([1, 2, 3], 2), ([[1, 2], [3, 4]], 3)],
)
def test_repeat_interleave(
    session: InferenceSession,
    device_ref: DeviceRef,
    dtype: DType,
    input: list[int] | list[list[int]],
    repeats: int,
) -> None:
    if device_ref.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    with Graph(
        "repeat_interleave",
        input_types=[],
    ) as graph:
        np_dtype = np.int64 if dtype == DType.int64 else np.float32
        x = ops.constant(
            np.array(input, dtype=np_dtype), dtype, device=device_ref
        )

        output = ops.repeat_interleave(x, repeats)
        graph.output(output)

    expected = (
        torch.repeat_interleave(
            torch.tensor(
                input,
                dtype=torch.int64 if dtype == DType.int64 else torch.float32,
            ),
            repeats,
        )
        .detach()
        .numpy()
    )

    model = session.load(graph)
    result = model.execute()[0]
    assert isinstance(result, Buffer)

    np.testing.assert_equal(result.to_numpy(), expected)


@pytest.mark.parametrize("device_ref", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input,repeats,axis",
    [
        # 1-d with matching length repeats
        ([1, 2, 3], [2, 0, 4], 0),
        # 1-d with broadcasted repeats
        ([1, 2, 3, 4], [4], 0),
        # 2-d along either axis
        ([[1, 2], [3, 4]], [1, 2], 0),
        ([[1, 2], [3, 4]], [1, 2], 1),
    ],
)
def test_repeat_interleave_vector(
    session: InferenceSession,
    device_ref: DeviceRef,
    input: list[int] | list[list[int]],
    repeats: list[int],
    axis: int,
) -> None:
    if device_ref.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    with Graph(
        "repeat_interleave_vector",
        input_types=[],
    ) as graph:
        x = ops.constant(input, DType.int64, device_ref)
        repeat_vals = ops.constant(repeats, DType.int64, DeviceRef.CPU())

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
    assert isinstance(result, Buffer)

    np.testing.assert_equal(result.to_numpy(), expected)


@pytest.mark.parametrize("device_ref", [DeviceRef.CPU(), DeviceRef.GPU()])
def test_repeat_interleave_runtime_input_float32(
    session: InferenceSession,
    device_ref: DeviceRef,
) -> None:
    if device_ref.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(DType.float32, [2, 3], device=device_ref)
    with Graph(
        "repeat_interleave_runtime_input_float32",
        input_types=[input_type],
    ) as graph:
        output = ops.repeat_interleave(graph.inputs[0].tensor, 2, axis=1)
        graph.output(output)

    model = session.load(graph)
    input_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = torch.repeat_interleave(
        torch.from_numpy(input_data),
        2,
        dim=1,
    ).numpy()

    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    np.testing.assert_array_equal(result.to_numpy(), expected)
