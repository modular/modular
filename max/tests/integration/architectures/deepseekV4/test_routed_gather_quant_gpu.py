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
"""``quantize_dynamic_scaled_float8_grouped_mx_gather`` against the chain it
replaces in the DeepSeek-V4 routed experts: ``ops.gather`` of the token rows
by slot, then ``quantize_dynamic_scaled_float8_grouped_mx``.

Every code, and every scale row a group's GEMM reads (the group's own rows of
its tiles, plus the zeroed rest of its last tile), must match bit for bit, on
every accelerator, at the decode and prefill row counts the model runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    quantize_dynamic_scaled_float8_grouped_mx,
    quantize_dynamic_scaled_float8_grouped_mx_gather,
)
from numpy.typing import NDArray

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the op is GPU only"
)

SF_ROWS = 128
K = 4096
TOPK = 6
GROUPS = 16


def _build(device: Accelerator) -> Model:
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "routed_gather_quant",
        input_types=[
            TensorType(DType.float32, ["tokens", K], ref),
            TensorType(DType.int32, ["slots"], ref),
            TensorType(DType.uint32, [GROUPS + 1], ref),
            TensorType(DType.uint32, [GROUPS], ref),
            TensorType(DType.int32, ["scale_rows"], ref),
        ],
    )
    with graph:
        x32, slot_token, starts, offsets, scale_slot = (
            v.tensor for v in graph.inputs
        )
        x = ops.cast(x32, DType.bfloat16)
        scale_rows = scale_slot.shape[0]
        codes, scales = quantize_dynamic_scaled_float8_grouped_mx_gather(
            x, slot_token, starts, offsets, scale_rows
        )
        ref_codes, ref_scales = quantize_dynamic_scaled_float8_grouped_mx(
            ops.gather(x, slot_token, axis=0), starts, offsets, scale_rows
        )
        graph.output(codes, scales, ref_codes, ref_scales)
    return InferenceSession(devices=[device]).load(graph)


def _sf_rows(mx: NDArray[np.uint8], rows: int) -> NDArray[np.uint8]:
    """5D ``[rows/128, K/128, 32, 4, 4]`` -> ``[rows, K/32]``."""
    tiles, kb = mx.shape[0], mx.shape[1]
    flat = mx.transpose(0, 3, 2, 1, 4).reshape(tiles * SF_ROWS, kb * 4)
    return flat[:rows]


@pytest.mark.parametrize("device_id", range(accelerator_count()))
@pytest.mark.parametrize("tokens", [1, 8, 16, 64, 2048])
def test_matches_gather_then_quantize(device_id: int, tokens: int) -> None:
    rng = np.random.default_rng(tokens * 7 + device_id)
    slots = tokens * TOPK
    counts = np.bincount(rng.integers(0, GROUPS, slots), minlength=GROUPS)
    starts = np.concatenate([[0], np.cumsum(counts)])
    aligned = (counts + SF_ROWS - 1) // SF_ROWS * SF_ROWS
    aligned_starts = np.concatenate([[0], np.cumsum(aligned)])
    offsets = aligned_starts[:-1] // SF_ROWS - starts[:-1] // SF_ROWS
    scale_rows = SF_ROWS * (-(-slots // SF_ROWS) + min(GROUPS, slots))
    assert scale_rows >= aligned_starts[-1]

    x = rng.normal(0.0, 4.0, (tokens, K)).astype(np.float32)
    x[0, :128] = 0.0  # a zero group: scale floor, zero recip
    slot_token = rng.integers(0, tokens, slots).astype(np.int32)
    scale_slot = np.zeros(scale_rows, dtype=np.int32)

    device = Accelerator(id=device_id)
    outs = _build(device).execute(
        Buffer.from_numpy(x).to(device),
        Buffer.from_numpy(slot_token).to(device),
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
    for g in range(GROUPS):
        a = int(aligned_starts[g])
        tile_end = a + int(aligned[g])
        np.testing.assert_array_equal(got[a:tile_end], want[a:tile_end], g)
