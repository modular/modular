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
"""``mo.quantize_dynamic_scaled_float8.grouped_mx`` against the graph chain
it replaces in the DeepSeek-V4 routed experts.

The chain: ``mo.quantize_dynamic_scaled_float8`` (1x128, e8m0), the scales
gathered by scale row (``scale_slot``), each repeated over its four 32-element
groups, then ``block_scales_interleave``. Every row of the fused op's codes and
every scale row a group's GEMM reads -- the group's own rows of its tiles --
must match the chain bit for bit; the rest of the group's last tile is zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    block_scales_interleave,
    quantize_dynamic_scaled_float8,
    quantize_dynamic_scaled_float8_grouped_mx,
)
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from numpy.typing import NDArray

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the op is GPU only"
)

SF_ROWS = 128


def _build(k: int, groups: int) -> Model:
    device = Accelerator()
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "w4a8_quant_rows",
        input_types=[
            TensorType(DType.float32, ["rows", k], ref),
            TensorType(DType.uint32, [groups + 1], ref),
            TensorType(DType.uint32, [groups], ref),
            TensorType(DType.int32, ["scale_rows"], ref),
        ],
    )
    with graph:
        x32, starts, offsets, scale_slot = (v.tensor for v in graph.inputs)
        x = ops.cast(x32, DType.bfloat16)
        scale_rows = scale_slot.shape[0]

        codes, scales = quantize_dynamic_scaled_float8_grouped_mx(
            x, starts, offsets, scale_rows
        )

        ref_codes, ref_scales = quantize_dynamic_scaled_float8(
            x,
            InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            group_size_or_per_token=128,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float8_e8m0fnu,
        )
        per_row = ops.transpose(ref_scales[:, 0 : x.shape[0]], 0, 1)
        per_row = ops.gather(per_row, scale_slot, axis=0)
        per_row = ops.reshape(
            ops.broadcast_to(
                ops.unsqueeze(per_row, -1), [scale_rows, k // 128, 4]
            ),
            [scale_rows, k // 32],
        )
        ref_mx = block_scales_interleave(per_row, 32)

        graph.output(codes, scales, ref_codes, ref_mx)
    return InferenceSession(devices=[device]).load(graph)


def _layout(
    counts: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    starts = np.concatenate([[0], np.cumsum(counts)])
    aligned = (counts + SF_ROWS - 1) // SF_ROWS * SF_ROWS
    aligned_starts = np.concatenate([[0], np.cumsum(aligned)])
    offsets = aligned_starts[:-1] // SF_ROWS - starts[:-1] // SF_ROWS
    return starts, aligned_starts, offsets


def _sf_rows(mx: NDArray[np.uint8], rows: int) -> NDArray[np.uint8]:
    """5D ``[rows/128, K/128, 32, 4, 4]`` -> ``[rows, K/32]``."""
    tiles, kb = mx.shape[0], mx.shape[1]
    # (tile, kb, r32, r4, k4) -> (tile, r4, r32, kb, k4): row = 32 * r4 + r32.
    flat = mx.transpose(0, 3, 2, 1, 4).reshape(tiles * SF_ROWS, kb * 4)
    return flat[:rows]


# (counts per local group, rows past the last local group).
CASES = {
    "decode_ep": ([0] * 40 + [1] + [0] * 50 + [2, 1] + [0] * 34 + [1], 1),
    "tile_crossing": ([300, 0, 129, 0, 1, 128, 0, 0], 0),
    "only_last": ([0, 0, 0, 5], 3),
    "single": ([257], 0),
}


@pytest.mark.parametrize("k", [4096, 2048])
@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_graph_chain(k: int, case: str) -> None:
    local, remote = CASES[case]
    counts = np.array(local, dtype=np.int64)
    groups = len(counts)
    starts, aligned_starts, offsets = _layout(counts)
    rows = int(starts[-1]) + remote
    # Scale rows as the model allocates them (tiles bounded by the non-empty
    # groups, at most min(groups, slots)); the remote rows' trailing group
    # counts as one more group.
    scale_rows = SF_ROWS * (-(-rows // SF_ROWS) + min(groups + 1, rows))
    assert scale_rows >= aligned_starts[-1]

    rng = np.random.default_rng(k + groups + rows)
    x = rng.normal(0.0, 4.0, (rows, k)).astype(np.float32)
    if rows > 2:
        x[1, :128] = 0.0  # a zero group: scale floor, zero recip
        x[2] *= 1e-30  # far below the e4m3 range
    scale_slot = np.zeros(scale_rows, dtype=np.int32)
    for g in range(groups):
        n = int(counts[g])
        a = int(aligned_starts[g])
        scale_slot[a : a + n] = np.arange(starts[g], starts[g] + n)

    model = _build(k, groups)
    device = Accelerator()
    outs = model.execute(
        Buffer.from_numpy(x).to(device),
        Buffer.from_numpy(starts.astype(np.uint32)).to(device),
        Buffer.from_numpy(offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(scale_slot).to(device),
    )
    codes, mx, ref_codes, ref_mx = (
        o.to(CPU()).view(DType.uint8).to_numpy() for o in outs
    )

    np.testing.assert_array_equal(codes, ref_codes)
    got = _sf_rows(mx, scale_rows)
    want = _sf_rows(ref_mx, scale_rows)
    for g in range(groups):
        n = int(counts[g])
        if n == 0:
            continue
        a = int(aligned_starts[g])
        np.testing.assert_array_equal(got[a : a + n], want[a : a + n])
        tile_end = a + (n + SF_ROWS - 1) // SF_ROWS * SF_ROWS
        assert not got[a + n : tile_end].any(), g
