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
"""Fused NVFP4 dequant-GEMV for GPUs without native FP4 matmul (pre-Blackwell).

Computes ``C[M, N] = A[M, K] @ W.T`` where ``W`` is ``[N, K // 2]`` packed FP4
(uint8, two E2M1 values per byte) with float32 block scales ``[N, K // 16]``
(modelopt ``weight_scale_2`` pre-multiplied by the caller). The weight is
decoded in registers and never materialized in global memory, so per-token
DRAM traffic is the packed bytes only (~0.5 B/element vs 4.5 B/element for
dequantize-then-matmul).

Marlin-style in spirit but GEMV-shaped: optimized for small M (decode);
activation rows are processed in tiles of ``M_TILE`` so a prefill of P tokens
re-reads the packed weight only ``ceil(P / M_TILE)`` times.
"""

from std.math import ceildiv
import std.gpu.primitives.warp as warp
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives.grid_controls import PDL
from layout import TileTensor
from layout.coord import Coord
from layout.tile_layout import TensorLayout

from .fp4_utils import decode_fp4e2m1_marlin, FP4E2M1_MARLIN_BIAS

comptime NVFP4_GEMV_SF_VECTOR_SIZE = 16
"""Elements covered by one NVFP4 block scale."""


@__name(t"nvfp4_gemv_{c_type}_{a_type}_{M_TILE}")
def _nvfp4_gemv_kernel[
    c_type: DType,
    a_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    w_layout: TensorLayout,
    s_layout: TensorLayout,
    *,
    M_TILE: Int = 4,
    WARPS_PER_BLOCK: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, MutAnyOrigin],
    w: TileTensor[DType.uint8, w_layout, MutAnyOrigin],
    scales: TileTensor[DType.float32, s_layout, MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """One warp owns one output column n; lanes stride K in 32-element chunks.

    Each chunk is one uint8x16 load (32 FP4 values = two scale blocks), one
    bf16x32 activation load per row in the M-tile, decode in registers, two
    scaled partial dots. Warp-reduce per row, lane 0 stores C[mm, n].
    """
    comptime CHUNK = 32
    comptime BYTES_PER_CHUNK = CHUNK // 2

    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var col = Int(block_idx.x) * WARPS_PER_BLOCK + warp_id
    var num_chunks = ceildiv(k, CHUNK)

    # All threads must enter the PDL region; tail-block threads with no
    # output column simply skip the work.
    with PDL():
        if col >= n:
            return
        for m_base in range(0, m, M_TILE):
            var acc = SIMD[DType.float32, M_TILE](0)

            for chunk_idx in range(lane, num_chunks, WARP_SIZE):
                var k_base = chunk_idx * CHUNK

                var packed = w.load[BYTES_PER_CHUNK](Coord(col, k_base // 2))
                # Marlin-style decode returns values at 2^-14 of the true
                # magnitude; the 2^14 bias is folded into the block scales below
                # (free -- the scale multiply happens anyway).
                var vals = decode_fp4e2m1_marlin(packed)
                var s0 = (
                    rebind[Scalar[DType.float32]](
                        scales.load(
                            Coord(col, k_base // NVFP4_GEMV_SF_VECTOR_SIZE)
                        )
                    )
                    * FP4E2M1_MARLIN_BIAS
                )
                var s1 = (
                    rebind[Scalar[DType.float32]](
                        scales.load(
                            Coord(col, k_base // NVFP4_GEMV_SF_VECTOR_SIZE + 1)
                        )
                    )
                    * FP4E2M1_MARLIN_BIAS
                )
                var w_lo = vals.slice[16, offset=0]()
                var w_hi = vals.slice[16, offset=16]()

                comptime for mi in range(M_TILE):
                    var mm = m_base + mi
                    if mm < m:
                        var xv = a.load[CHUNK](Coord(mm, k_base)).cast[
                            DType.float32
                        ]()
                        # Scales are constant per block: factor them out of
                        # the dot products.
                        acc[mi] += (
                            s0 * (w_lo * xv.slice[16, offset=0]()).reduce_add()
                            + s1
                            * (w_hi * xv.slice[16, offset=16]()).reduce_add()
                        )

            comptime for mi in range(M_TILE):
                var total = warp.sum(acc[mi])
                if lane == 0 and m_base + mi < m:
                    c.store(
                        Coord(m_base + mi, col),
                        SIMD[c_type, 1](total.cast[c_type]()),
                    )


@always_inline
def nvfp4_gemv(
    ctx: DeviceContext,
    c: TileTensor,
    a: TileTensor,
    w_packed: TileTensor,
    scales: TileTensor,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Launches the fused NVFP4 dequant-GEMV.

    Args:
        ctx: Device context for the launch.
        c: Output [m, n], bfloat16 or float32.
        a: Activations [m, k], bfloat16.
        w_packed: Packed FP4 weight [n, k // 2], uint8.
        scales: Float32 block scales [n, k // 16] (per-tensor scale
            pre-multiplied).
        m: Number of activation rows.
        n: Output features (weight rows).
        k: Inner dimension (unpacked element count, multiple of 32).
    """
    comptime assert scales.dtype == DType.float32, "scales must be float32"
    comptime assert (
        w_packed.dtype == DType.uint8
    ), "packed weights must be uint8"
    if k % 32 != 0:
        raise Error("nvfp4_gemv requires k to be a multiple of 32")

    comptime WARPS_PER_BLOCK = 4
    comptime M_TILE = 4

    comptime kernel = _nvfp4_gemv_kernel[
        c.dtype,
        a.dtype,
        c.LayoutType,
        a.LayoutType,
        w_packed.LayoutType,
        scales.LayoutType,
        M_TILE=M_TILE,
        WARPS_PER_BLOCK=WARPS_PER_BLOCK,
    ]

    var c_dev = rebind[TileTensor[c.dtype, c.LayoutType, MutAnyOrigin]](c)
    var a_dev = rebind[TileTensor[a.dtype, a.LayoutType, MutAnyOrigin]](a)
    var w_dev = rebind[
        TileTensor[DType.uint8, w_packed.LayoutType, MutAnyOrigin]
    ](w_packed)
    var s_dev = rebind[
        TileTensor[DType.float32, scales.LayoutType, MutAnyOrigin]
    ](scales)

    ctx.enqueue_function[kernel](
        c_dev,
        a_dev,
        w_dev,
        s_dev,
        m,
        n,
        k,
        block_dim=(WARPS_PER_BLOCK * WARP_SIZE, 1, 1),
        grid_dim=(ceildiv(n, WARPS_PER_BLOCK), 1, 1),
    )
