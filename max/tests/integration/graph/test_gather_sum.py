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

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import gather
from max.graph.ops import sum as reduce_sum


def _run_gather_sum(
    session: InferenceSession,
    table: torch.Tensor,
    indices: torch.Tensor,
) -> np.ndarray:
    """Builds gather(axis=0) + sum(axis=1), which the compiler fuses to
    mo.gather_sum, and returns the executed result."""
    device_to_use = DeviceRef.from_device(session.devices[0])
    with Graph(
        "gather_sum",
        input_types=[
            TensorType(DType.float32, list(table.shape), device=device_to_use),
            TensorType(DType.int32, list(indices.shape), device=device_to_use),
        ],
    ) as graph:
        table_value, indices_value = graph.inputs

        gathered = gather(table_value.tensor, indices_value.tensor, axis=0)
        output = reduce_sum(gathered, axis=1)
        graph.output(output)

    model = session.load(graph)

    actual_tensor = model(
        Buffer.from_dlpack(table).to(model.input_devices[0]),
        Buffer.from_dlpack(indices).to(model.input_devices[1]),
    )[0]
    assert isinstance(actual_tensor, Buffer)
    return actual_tensor.to_numpy()


def test_gather_sum_on_cpu() -> None:
    """Fused multi-hot embedding sums run on the host."""
    # A dedicated CPU session: the shared fixture prefers accelerators.
    session = InferenceSession(devices=[CPU()])
    table = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)

    actual = _run_gather_sum(session, table, indices)

    # EmbeddingBag-style multi-hot sums: one row per batch element.
    expected = table[indices.numpy()].sum(axis=1)
    np.testing.assert_allclose(actual.reshape(2, 3), expected)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="gather_sum GPU execution requires an accelerator",
)
def test_gather_sum_on_gpu(session: InferenceSession) -> None:
    """A device-resident embedding table no longer round-trips through the
    host: the fused gather_sum executes on the input's device (MXF-821)."""
    table = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    indices = torch.tensor([[0, 3, 9], [1, 1, 1], [7, 0, 3]], dtype=torch.int32)

    actual = _run_gather_sum(session, table, indices)

    expected = table[indices.numpy()].sum(axis=1)
    np.testing.assert_allclose(actual.reshape(3, 6), expected)
