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
"""Elementwise loops on GPU whose innermost extent the vector width doesn't
divide, so `elementwise` takes its uneven path."""

import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


# Inputs count through the flat index modulo 256, so bf16 holds every value
# exactly and a misread or shifted lane changes the result.
@pytest.mark.skipif(accelerator_count() == 0, reason="requires an accelerator")
@pytest.mark.parametrize(
    "dtype, torch_dtype, shape",
    [
        # Rows of 12 at width 8 run at 4 lanes, which divide the rows.
        (DType.float32, torch.float32, [7, 12]),
        # Rows of 100 at width 16 keep the width; loads are aligned to 4.
        (DType.bfloat16, torch.bfloat16, [5, 100]),
        # Rank 3, rows of 6 at width 8: 4 lanes, loads aligned to 2.
        (DType.float32, torch.float32, [3, 4, 6]),
        # Odd rows hit the 4-lane floor, so vectors still cross row ends.
        (DType.float32, torch.float32, [9, 5]),
        (DType.float32, torch.float32, [3, 4, 7]),
        # Rows of 6 at width 16: the largest dividing power of two, 2, is
        # raised to 4.
        (DType.bfloat16, torch.bfloat16, [9, 6]),
    ],
)
def test_elementwise_uneven_rows(
    session: InferenceSession,
    dtype: DType,
    torch_dtype: torch.dtype,
    shape: list[int],
) -> None:
    device = DeviceRef.from_device(session.devices[0])
    with Graph(
        "elementwise_uneven_rows",
        input_types=[TensorType(dtype, shape, device=device)],
    ) as graph:
        (x,) = graph.inputs
        graph.output(x.tensor + x.tensor)

    model = session.load(graph)
    numel = 1
    for d in shape:
        numel *= d
    x_t = (torch.arange(numel) % 256).reshape(shape).to(torch_dtype)
    (result,) = model.execute(
        Buffer.from_dlpack(x_t).to(model.input_devices[0])
    )
    assert isinstance(result, Buffer)
    torch.testing.assert_close(
        torch.from_dlpack(result).cpu(), x_t + x_t, rtol=0, atol=0
    )
