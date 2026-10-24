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
that live count. Each row's values must match the first ``k[r]`` values of
the full top-k bit for bit, and the rest must be ``(-inf, -1)``. Equal scores
may come out in any order, so columns are only checked to be unique and to
index the value in their slot. The widths cover the full 2048 and 1030, which
pads the bitonic sort's tail and puts odd rows' bases off a 16 byte boundary.

A pick count can also exceed a row's finite entries, as the indexer's would
without its ``min(live, k)``: the picked range then carries ``-inf`` slots
whose column may be a sort padding lane (negative).

The top-k does not accept NaN, so the indexer masks it before selecting. The
last test checks a NaN score neither gets picked nor displaces a real one.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import top_k_per_row
from max.pipelines.architectures.deepseekV4.layers.indexer import (
    DeepseekV4Indexer,
)
from numpy.typing import NDArray

N = 2048
MAX_K = 512
WIDTHS = (2048, 1030)

Compiled = tuple[Device, Model, int]


def _devices() -> list[str]:
    return ["cpu", "gpu"] if accelerator_count() > 0 else ["cpu"]


@pytest.fixture(
    scope="module",
    params=[(d, n) for d in _devices() for n in WIDTHS],
    ids=lambda p: f"{p[0]}-n{p[1]}",
)
def compiled(request: pytest.FixtureRequest) -> Compiled:
    name, n = request.param
    device: Device = CPU() if name == "cpu" else Accelerator()
    ref = DeviceRef.from_device(device)
    graph = Graph(
        f"top_k_per_row_{n}",
        input_types=[
            TensorType(DType.float32, ["rows", n], ref),
            TensorType(DType.int64, ["rows"], ref),
        ],
    )
    with graph:
        x, k = (v.tensor for v in graph.inputs)
        full_vals, full_idxs = ops.top_k(x, MAX_K, axis=-1)
        vals, idxs = top_k_per_row(x, k, MAX_K)
        graph.output(full_vals, full_idxs, vals, idxs)
    session = InferenceSession(devices=[device])
    return device, session.load(graph), n


def _scores(
    rng: np.random.Generator, n: int, live: NDArray[np.int64], ties: bool
) -> NDArray[np.float32]:
    rows = len(live)
    if ties:
        # Few distinct values, so most picks are decided by the tie-break.
        x = rng.integers(0, 8, (rows, n)).astype(np.float32)
    else:
        x = rng.normal(0.0, 1.0, (rows, n)).astype(np.float32)
    # The indexer's layout: live candidates are a zone prefix plus a few
    # fresh windows at the far end, everything else -inf.
    for r in range(rows):
        zone = int(live[r]) - int(live[r]) // 8
        dead = np.ones(n, dtype=bool)
        dead[:zone] = False
        dead[n // 2 : n // 2 + int(live[r]) - zone] = False
        x[r, dead] = -np.inf
    return x


@pytest.mark.parametrize("rows", [1, 7, 200])
@pytest.mark.parametrize("ties", [False, True])
def test_matches_full_top_k_prefix(
    compiled: Compiled, rows: int, ties: bool
) -> None:
    device, model, n = compiled
    rng = np.random.default_rng(rows * 2 + ties)
    live = rng.integers(0, n // 2, rows).astype(np.int64)
    live[0] = 0
    if rows > 1:
        live[1] = n // 2
    x = _scores(rng, n, live, ties)
    k = np.minimum(live, MAX_K)

    outs = model.execute(
        Buffer.from_numpy(x).to(device), Buffer.from_numpy(k).to(device)
    )
    full_vals, _, vals, idxs = (o.to(CPU()).to_numpy() for o in outs)

    for r in range(rows):
        kr = int(k[r])
        np.testing.assert_array_equal(
            vals[r, :kr].view(np.uint32), full_vals[r, :kr].view(np.uint32)
        )
        cols = idxs[r, :kr]
        assert np.all((cols >= 0) & (cols < n)), r
        assert len(np.unique(cols)) == kr, r
        np.testing.assert_array_equal(
            x[r, cols].view(np.uint32), vals[r, :kr].view(np.uint32)
        )
        assert np.all(vals[r, kr:] == -np.inf), r
        assert np.all(idxs[r, kr:] == -1), r


@pytest.mark.parametrize("ties", [False, True])
def test_picks_past_finite_entries(compiled: Compiled, ties: bool) -> None:
    device, model, n = compiled
    rng = np.random.default_rng(100 + ties)
    live = np.minimum(
        np.array([0, 1, 62, 511, 512, 513, n // 2], dtype=np.int64), n
    )
    x = _scores(rng, n, live, ties)
    k = np.full(len(live), MAX_K, dtype=np.int64)

    outs = model.execute(
        Buffer.from_numpy(x).to(device), Buffer.from_numpy(k).to(device)
    )
    full_vals, _, vals, idxs = (o.to(CPU()).to_numpy() for o in outs)

    for r in range(len(live)):
        np.testing.assert_array_equal(
            vals[r].view(np.uint32), full_vals[r].view(np.uint32)
        )
        cols = idxs[r]
        assert np.all(cols < n), r
        real = cols >= 0
        assert len(np.unique(cols[real])) == int(real.sum()), r
        np.testing.assert_array_equal(
            x[r, cols[real]].view(np.uint32), vals[r, real].view(np.uint32)
        )
        assert np.all(vals[r, ~real] == -np.inf), r


@pytest.fixture(scope="module", params=_devices())
def indexer_select(request: pytest.FixtureRequest) -> tuple[Device, Model]:
    device: Device = CPU() if request.param == "cpu" else Accelerator()
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "indexer_select",
        input_types=[
            TensorType(DType.float32, ["rows", N], ref),
            TensorType(DType.bool, ["rows", N], ref),
        ],
    )
    with graph:
        score, valid = (v.tensor for v in graph.inputs)
        # select only reads index_topk from the layer.
        layer = Mock(spec=DeepseekV4Indexer)
        layer.index_topk = MAX_K
        graph.output(DeepseekV4Indexer.select(layer, score, valid))
    session = InferenceSession(devices=[device])
    return device, session.load(graph)


def test_indexer_select_skips_nan(
    indexer_select: tuple[Device, Model],
) -> None:
    device, model = indexer_select
    rng = np.random.default_rng(0)
    rows = 4
    x = rng.normal(0.0, 1.0, (rows, N)).astype(np.float32)
    bits = x.view(np.uint32)
    # Positive quiet NaN ranks above +inf by bit pattern, and all ones is
    # what the block-wide top-k pads with.
    bits[:, 0::7] = 0x7FC00000
    bits[:, 3::11] = 0xFFFFFFFF
    valid = np.ones((rows, N), dtype=bool)
    valid[1, N // 4 :] = False
    valid[2, 600:] = False

    outs = model.execute(
        Buffer.from_numpy(x).to(device), Buffer.from_numpy(valid).to(device)
    )
    idxs = outs[0].to(CPU()).to_numpy()

    masked = np.where(valid & ~np.isnan(x), x, -np.inf)
    for r in range(rows):
        picks = min(int(valid[r].sum()), MAX_K)
        finite = int(np.isfinite(masked[r]).sum())
        expected = np.argsort(-masked[r], kind="stable")[: min(picks, finite)]
        got = idxs[r][idxs[r] >= 0]
        assert sorted(got.tolist()) == sorted(expected.tolist()), r
