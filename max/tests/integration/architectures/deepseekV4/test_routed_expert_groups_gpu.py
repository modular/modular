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
"""Every routed slot lands in a group ``moe_create_indices`` writes.

``moe_create_indices`` skips an expert id outside ``[0, groups)`` and leaves
that slot's order/restore entries unwritten, and ``DeepseekV4RoutedExperts``
gathers through them. GPU ``top_k`` returns -1 for a NaN score row, so such
ids do reach the MoE..
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import moe_create_indices
from max.pipelines.architectures.deepseekV4.layers.moe import (
    local_expert_groups,
)
from numpy.typing import NDArray

N_GLOBAL = 16
# Ids on both halves of a two-way expert split, then ones a NaN score row or
# a bad table entry produces.
IDS = np.array(
    [0, 3, 7, 8, 12, 15, 5, -1, 9, 16, 2**30, 7, 7, -1, 0], dtype=np.int32
)


def _run(
    build: Callable[[TensorValue], Sequence[TensorValue]],
) -> list[NDArray[np.generic]]:
    """Runs ``build`` over :data:`IDS` on the GPU."""
    ids_type = TensorType(DType.int32, ["slots"], DeviceRef.GPU())
    with Graph("expert_ids", input_types=[ids_type]) as graph:
        graph.output(*build(graph.inputs[0].tensor))
    device = Accelerator()
    model = InferenceSession(devices=[device]).load(graph)
    arrays = []
    for out in model.execute(Buffer.from_numpy(IDS).to(device)):
        assert isinstance(out, Buffer)
        arrays.append(out.to_numpy())
    return arrays


@pytest.mark.parametrize("first, n_local", [(0, N_GLOBAL), (0, 8), (8, 8)])
def test_every_slot_is_grouped(first: int, n_local: int) -> None:
    def build(ids: TensorValue) -> list[TensorValue]:
        grouped, is_local = local_expert_groups(ids, first, n_local)
        order, start, restore, _, _ = moe_create_indices(grouped, n_local + 1)
        return [grouped, is_local, order, start, restore]

    grouped, is_local, order, start, restore = _run(build)

    local = (IDS >= first) & (IDS < first + n_local)
    np.testing.assert_array_equal(is_local, local)
    np.testing.assert_array_equal(
        grouped, np.where(local, IDS - first, n_local)
    )
    slots = len(IDS)
    assert start[n_local + 1] == slots
    assert start[n_local + 1] - start[n_local] == np.count_nonzero(~local)
    np.testing.assert_array_equal(np.sort(order), np.arange(slots))
    np.testing.assert_array_equal(order[restore], np.arange(slots))


def test_index_kernel_skips_out_of_range_ids() -> None:
    """The hazard ``local_expert_groups`` closes: raw ids leave slots out."""
    (start,) = _run(lambda ids: [moe_create_indices(ids, N_GLOBAL)[1]])

    in_range = np.count_nonzero((IDS >= 0) & (IDS < N_GLOBAL))
    assert start[N_GLOBAL] == in_range < len(IDS)
