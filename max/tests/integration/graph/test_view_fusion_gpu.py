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

"""GPU end-to-end test for view fusion alignment.

Split from `test_view_fusion.py`, whose graphs all run on CPU, so that only
this test holds a GPU executor.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_inner_slice_misaligned_row_stride_fuses(
    session: InferenceSession,
) -> None:
    """An inner-dim slice of a 2-byte-dtype tensor whose row stride is not a
    multiple of the GPU vector width, fused into an elementwise op, must not
    over-report the view's alignment and emit a misaligned (faulting) vector
    load.

    Regression for the gated-deltanet gate crash
    (``CUDA_ERROR_MISALIGNED_ADDRESS``): ``_mogg_slice_view_alignment`` dropped
    the outer-dim stride term that ``Slice.get_view_alignment`` carries. A
    ``[8, 1030]`` matmul output has a 2060-byte row stride (1030 is not a
    multiple of the 16-element / 32-byte float16 vector width), so its rows are
    only 4-byte aligned. Slicing it on the inner dim starting at ``(0, 0)``
    leaves the start term a no-op, so *only* the outer-stride term keeps the
    view's alignment honest: without it the view kept the buffer's full base
    alignment and the fused cast issued an over-aligned vector load that faulted
    on row 1. The slice reads the matmul output (a real, preferred-aligned
    buffer), and the cast fuses the slice into the elementwise that loads it.
    """
    n, k, d = 8, 256, 1030
    with Graph(
        "inner_slice_misaligned",
        input_types=[
            TensorType(DType.float16, [n, k], device=DeviceRef.GPU()),
            TensorType(DType.float16, [d, k], device=DeviceRef.GPU()),
        ],
    ) as graph:
        x, w = (v.tensor for v in graph.inputs)
        y = x @ ops.transpose(w, 0, 1)  # materialized [8, 1030] intermediate
        graph.output(ops.cast(y[:, 0:1024], DType.float32))

    # 0/1 inputs keep the float16 matmul exact (each row sum <= 256 < 2048).
    x_np = (np.random.rand(n, k) < 0.5).astype(np.float16)
    w_np = (np.random.rand(d, k) < 0.5).astype(np.float16)
    (out,) = run_and_verify_fusion(session, graph, x_np, w_np)
    ref = x_np.astype(np.float32) @ w_np.astype(np.float32).T
    np.testing.assert_array_equal(out, ref[:, 0:1024])
