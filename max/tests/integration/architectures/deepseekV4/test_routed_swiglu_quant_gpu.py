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
"""The DeepSeek-V4 routed SwiGLU reading its gate and up halves as slices of
the W4A8 gate/up GEMM output, against the ``ops.split`` it replaces, each
feeding ``quantize_dynamic_scaled_float8_grouped_mx``.

With both outputs of the split cast to float32, the graph compiler casts the
whole GEMM output once instead, a kernel of its own whose output has two
readers; from slices, the SwiGLU fuses into the quantize's input. The values
are the same, so every code, and every scale row a group's GEMM reads, must
match bit for bit, on every accelerator, at the decode and prefill token
counts the model runs. The two run as separate graphs, since the rewrite
needs every user of the split to be a cast.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn._graph_ops import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    quantize_dynamic_scaled_float8_grouped_mx,
)
from numpy.typing import NDArray

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the op is GPU only"
)

SF_ROWS = 128
HIDDEN = 4096
INTER = 2048
TOPK = 6
GROUPS = 16
LIMIT = 10.0

Halves = Callable[[TensorValue], tuple[TensorValue, TensorValue]]


def _split(gate_up: TensorValue) -> tuple[TensorValue, TensorValue]:
    gate, up = ops.split(gate_up, [INTER, INTER], axis=1)
    return gate, up


def _slices(gate_up: TensorValue) -> tuple[TensorValue, TensorValue]:
    return gate_up[:, :INTER], gate_up[:, INTER:]


def _build(device: Accelerator, halves: Halves) -> Model:
    """``_routed_body`` from the routed rows to the down quantize."""
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "routed_swiglu_quant",
        input_types=[
            TensorType(DType.float32, ["slots", HIDDEN], ref),
            TensorType(DType.uint8, [GROUPS, 2 * INTER, HIDDEN // 2], ref),
            TensorType(
                DType.float8_e8m0fnu, [GROUPS, 2 * INTER, HIDDEN // 32], ref
            ),
            TensorType(DType.float32, ["slots"], ref),
            TensorType(DType.uint32, [GROUPS + 1], ref),
            TensorType(DType.uint32, [GROUPS], ref),
            TensorType(DType.int32, [GROUPS], ref),
            TensorType(DType.int32, ["scale_rows"], ref),
        ],
    )
    with graph:
        x, w, w_scale, weight, starts, offsets, expert_ids, scale_slot = (
            v.tensor for v in graph.inputs
        )
        scale_rows = scale_slot.shape[0]
        codes, a_scales = quantize_dynamic_scaled_float8_grouped_mx(
            ops.cast(x, DType.bfloat16), starts, offsets, scale_rows
        )
        b_scales = ops.stack(
            [
                block_scales_interleave(
                    ops.reshape(w_scale[e : e + 1], [2 * INTER, HIDDEN // 32]),
                    32,
                )
                for e in range(GROUPS)
            ],
            axis=0,
        )
        usage = ops.concat(
            [
                ops.cast(ops.shape_to_tensor([x.shape[0]]), DType.uint32),
                ops.constant(
                    np.array([GROUPS], np.uint32),
                    DType.uint32,
                    device=DeviceRef.CPU(),
                ),
            ],
            axis=0,
        )
        gate_up = grouped_matmul_block_scaled(
            codes,
            w,
            a_scales,
            b_scales,
            starts,
            offsets,
            expert_ids,
            ops.constant(np.ones(GROUPS, np.float32), DType.float32, ref),
            usage,
        )
        gate, up = halves(gate_up)
        gate = ops.cast(gate, DType.float32)
        up = ops.cast(up, DType.float32)
        limit = ops.constant(LIMIT, DType.float32, device=ref)
        up = ops.min(ops.max(up, -limit), limit)
        gate = ops.min(gate, limit)
        h = ops.silu(gate) * up * ops.unsqueeze(weight, -1)
        graph.output(
            gate_up,
            *quantize_dynamic_scaled_float8_grouped_mx(
                ops.cast(h, DType.bfloat16), starts, offsets, scale_rows
            ),
        )
    return InferenceSession(devices=[device]).load(graph)


def _sf_rows(mx: NDArray[np.uint8], rows: int) -> NDArray[np.uint8]:
    """5D ``[rows/128, K/128, 32, 4, 4]`` -> ``[rows, K/32]``."""
    tiles, kb = mx.shape[0], mx.shape[1]
    flat = mx.transpose(0, 3, 2, 1, 4).reshape(tiles * SF_ROWS, kb * 4)
    return flat[:rows]


@pytest.mark.parametrize("device_id", range(accelerator_count()))
@pytest.mark.parametrize("tokens", [1, 8, 16, 64, 2048])
def test_slices_match_split(device_id: int, tokens: int) -> None:
    rng = np.random.default_rng(tokens * 11 + device_id)
    slots = tokens * TOPK
    counts = np.bincount(rng.integers(0, GROUPS, slots), minlength=GROUPS)
    starts = np.concatenate([[0], np.cumsum(counts)])
    aligned = (counts + SF_ROWS - 1) // SF_ROWS * SF_ROWS
    aligned_starts = np.concatenate([[0], np.cumsum(aligned)])
    offsets = aligned_starts[:-1] // SF_ROWS - starts[:-1] // SF_ROWS
    scale_rows = SF_ROWS * (-(-slots // SF_ROWS) + min(GROUPS, slots))

    x = rng.normal(0.0, 1.0, (slots, HIDDEN)).astype(np.float32)
    w = rng.integers(0, 256, (GROUPS, 2 * INTER, HIDDEN // 2), np.uint8)
    # Power-of-two scales near 1/16: gate/up of order 10, so both clamps bind.
    w_scale = rng.integers(
        122, 125, (GROUPS, 2 * INTER, HIDDEN // 32), np.uint8
    )
    weight = rng.uniform(0.0, 1.0, slots).astype(np.float32)

    device = Accelerator(id=device_id)
    args = [
        Buffer.from_numpy(x),
        Buffer.from_numpy(w),
        Buffer.from_numpy(w_scale).view(DType.float8_e8m0fnu),
        Buffer.from_numpy(weight),
        Buffer.from_numpy(starts.astype(np.uint32)),
        Buffer.from_numpy(offsets.astype(np.uint32)),
        Buffer.from_numpy(np.arange(GROUPS, dtype=np.int32)),
        Buffer.from_numpy(np.zeros(scale_rows, np.int32)),
    ]
    gate_up, codes, mx, ref_gate_up, ref_codes, ref_mx = (
        o.to(CPU()).view(DType.uint8).to_numpy()
        for halves in (_slices, _split)
        for o in _build(device, halves).execute(*(a.to(device) for a in args))
    )

    # The same GEMM output into both, with values past both clamps.
    np.testing.assert_array_equal(gate_up, ref_gate_up)
    values = gate_up.view(np.uint16).astype(np.uint32) << np.uint32(16)
    assert np.abs(values.view(np.float32)).max() > LIMIT
    np.testing.assert_array_equal(codes, ref_codes)
    got = _sf_rows(mx, scale_rows)
    want = _sf_rows(ref_mx, scale_rows)
    for g in range(GROUPS):
        a = int(aligned_starts[g])
        np.testing.assert_array_equal(
            got[a : a + int(aligned[g])], want[a : a + int(aligned[g])], g
        )
