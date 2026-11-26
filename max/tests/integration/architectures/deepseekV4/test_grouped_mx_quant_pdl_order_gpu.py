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
"""``quantize_dynamic_scaled_float8_grouped_mx`` with its group tables written
by the launch right before it.

The quantizer runs under programmatic dependent launch, so it may start before
the grid that writes ``group_starts`` and ``scale_offsets`` has finished; every
read of those tables has to come after its grid-dependency wait. Here the
tables are computed in the graph from the group sizes, immediately before the
quantize, and the inputs change between runs so a stale read cannot match. The
codes and every scale row a group's GEMM reads must equal the same op fed the
tables as graph inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import quantize_dynamic_scaled_float8_grouped_mx
from numpy.typing import NDArray

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="the op is GPU only"
)

SF_ROWS = 128
K = 4096
GROUPS = 16
RUNS = 24


def _build(device: Accelerator, tables_in_graph: bool) -> Model:
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "grouped_mx_quant_pdl_order",
        input_types=[
            TensorType(DType.float32, ["rows", K], ref),
            TensorType(DType.int32, [GROUPS], ref),
            TensorType(DType.uint32, [GROUPS + 1], ref),
            TensorType(DType.uint32, [GROUPS], ref),
            TensorType(DType.int32, ["scale_rows"], ref),
        ],
    )
    with graph:
        x32, counts, starts_in, offsets_in, scale_slot = (
            v.tensor for v in graph.inputs
        )
        x = ops.cast(x32, DType.bfloat16)
        if tables_in_graph:
            # The tables come out of the launch right before the quantize.
            zero = ops.constant(np.zeros(1, np.int32), DType.int32, ref)
            starts = ops.concat([zero, ops.cumsum(counts, axis=0)], axis=0)
            aligned = (counts + (SF_ROWS - 1)) // SF_ROWS * SF_ROWS
            aligned_starts = ops.concat(
                [zero, ops.cumsum(aligned, axis=0)], axis=0
            )
            offsets = (
                aligned_starts[:GROUPS] // SF_ROWS - starts[:GROUPS] // SF_ROWS
            )
            starts = ops.cast(starts, DType.uint32)
            offsets = ops.cast(offsets, DType.uint32)
        else:
            starts, offsets = starts_in, offsets_in
        codes, scales = quantize_dynamic_scaled_float8_grouped_mx(
            x, starts, offsets, scale_slot.shape[0]
        )
        graph.output(codes, scales)
    return InferenceSession(devices=[device]).load(graph)


def _sf_rows(mx: NDArray[np.uint8], rows: int) -> NDArray[np.uint8]:
    """5D ``[rows/128, K/128, 32, 4, 4]`` -> ``[rows, K/32]``."""
    tiles, kb = mx.shape[0], mx.shape[1]
    flat = mx.transpose(0, 3, 2, 1, 4).reshape(tiles * SF_ROWS, kb * 4)
    return flat[:rows]


@pytest.mark.parametrize("device_id", range(accelerator_count()))
@pytest.mark.parametrize("rows", [96, 3072])
def test_tables_from_the_preceding_launch(device_id: int, rows: int) -> None:
    device = Accelerator(id=device_id)
    raced, plain = _build(device, True), _build(device, False)
    rng = np.random.default_rng(rows * 3 + device_id)
    for run in range(RUNS):
        # Different group sizes every run: a stale table never matches.
        counts = np.bincount(rng.integers(0, GROUPS, rows), minlength=GROUPS)
        starts = np.concatenate([[0], np.cumsum(counts)])
        aligned = (counts + SF_ROWS - 1) // SF_ROWS * SF_ROWS
        aligned_starts = np.concatenate([[0], np.cumsum(aligned)])
        offsets = aligned_starts[:-1] // SF_ROWS - starts[:-1] // SF_ROWS
        scale_rows = SF_ROWS * (-(-rows // SF_ROWS) + min(GROUPS, rows))
        x = rng.normal(0.0, 4.0, (rows, K)).astype(np.float32)
        args = [
            Buffer.from_numpy(x).to(device),
            Buffer.from_numpy(counts.astype(np.int32)).to(device),
            Buffer.from_numpy(starts.astype(np.uint32)).to(device),
            Buffer.from_numpy(offsets.astype(np.uint32)).to(device),
            Buffer.from_numpy(np.zeros(scale_rows, np.int32)).to(device),
        ]
        codes, mx = (
            o.to(CPU()).view(DType.uint8).to_numpy()
            for o in raced.execute(*args)
        )
        ref_codes, ref_mx = (
            o.to(CPU()).view(DType.uint8).to_numpy()
            for o in plain.execute(*args)
        )
        np.testing.assert_array_equal(codes, ref_codes, f"run {run}")
        got, want = _sf_rows(mx, scale_rows), _sf_rows(ref_mx, scale_rows)
        for g in range(GROUPS):
            a = int(aligned_starts[g])
            np.testing.assert_array_equal(
                got[a : a + int(aligned[g])],
                want[a : a + int(aligned[g])],
                f"run {run} group {g}",
            )
