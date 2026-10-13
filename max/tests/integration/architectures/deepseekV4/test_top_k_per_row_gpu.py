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
"""``mo.top_k.per_row`` against ``mo.top_k`` in the same graph.

The DeepSeek-V4 indexer's shape: 2048 candidates, 512 kept, a masked tail of
``-inf`` past each row's live candidates, and a pick count per row equal to
that live count. Each row must reproduce the first ``k[r]`` picks of the full
top-k bit for bit, ties included, and pad the rest with ``(-inf, -1)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import top_k_per_row
from numpy.typing import NDArray

N = 2048
MAX_K = 512


def _devices() -> list[str]:
    return ["cpu", "gpu"] if accelerator_count() > 0 else ["cpu"]


@pytest.fixture(scope="module", params=_devices())
def compiled(request: pytest.FixtureRequest) -> tuple[Device, Model]:
    device: Device = CPU() if request.param == "cpu" else Accelerator()
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "top_k_per_row",
        input_types=[
            TensorType(DType.float32, ["rows", N], ref),
            TensorType(DType.int64, ["rows"], ref),
        ],
    )
    with graph:
        x, k = (v.tensor for v in graph.inputs)
        full_vals, full_idxs = ops.top_k(x, MAX_K, axis=-1)
        vals, idxs = top_k_per_row(x, k, MAX_K)
        graph.output(full_vals, full_idxs, vals, idxs)
    session = InferenceSession(devices=[device])
    return device, session.load(graph)


def _scores(
    rng: np.random.Generator, rows: int, live: NDArray[np.int64], ties: bool
) -> NDArray[np.float32]:
    if ties:
        # Few distinct values, so most picks are decided by the tie-break.
        x = rng.integers(0, 8, (rows, N)).astype(np.float32)
    else:
        x = rng.normal(0.0, 1.0, (rows, N)).astype(np.float32)
    # The indexer's layout: live candidates are a zone prefix plus a few
    # fresh windows at the far end, everything else -inf.
    for r in range(rows):
        zone = int(live[r]) - int(live[r]) // 8
        dead = np.ones(N, dtype=bool)
        dead[:zone] = False
        dead[N // 2 : N // 2 + int(live[r]) - zone] = False
        x[r, dead] = -np.inf
    return x


@pytest.mark.parametrize("rows", [1, 7, 200])
@pytest.mark.parametrize("ties", [False, True])
def test_matches_full_top_k_prefix(
    compiled: tuple[Device, Model], rows: int, ties: bool
) -> None:
    device, model = compiled
    rng = np.random.default_rng(rows * 2 + ties)
    live = rng.integers(0, N // 2, rows).astype(np.int64)
    live[0] = 0
    if rows > 1:
        live[1] = N // 2
    x = _scores(rng, rows, live, ties)
    k = np.minimum(live, MAX_K)

    outs = model.execute(
        Buffer.from_numpy(x).to(device), Buffer.from_numpy(k).to(device)
    )
    full_vals, full_idxs, vals, idxs = (o.to(CPU()).to_numpy() for o in outs)

    for r in range(rows):
        kr = int(k[r])
        np.testing.assert_array_equal(
            vals[r, :kr].view(np.uint32), full_vals[r, :kr].view(np.uint32)
        )
        np.testing.assert_array_equal(idxs[r, :kr], full_idxs[r, :kr])
        assert np.all(vals[r, kr:] == -np.inf), r
        assert np.all(idxs[r, kr:] == -1), r
