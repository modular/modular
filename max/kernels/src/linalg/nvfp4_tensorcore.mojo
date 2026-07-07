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
"""Native consumer-Blackwell (sm_120 / sm_121) NVFP4 tensor-core GEMM.

Uses the warp-level block-scaled FP4 tensor-core MMA
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X`,
which is available on consumer Blackwell (RTX 50xx / DGX Spark GB10) and is the
performant alternative to the arch-agnostic naive CUDA-core path. One warp per
16x8 output tile, grid = (ceil(N/8), ceil(M/16)), K-loop over 64-element chunks.

Scales are read from MAX's rank-5 interleaved SF layout
`[ceil(rows/128), ceil(K/64), 32, 4, 4]` (e4m3) via `sf_rank5_off` — the same
mapping `set_scale_factor` writes (fp4_utils.mojo). A K-group's 4 blocks are
contiguous, so the four e4m3 scales load as one UInt32, matching the MMA's
`scale_vec::4X` per-lane distribution.

Requirements: target `sm_121a` (plain `sm_121` yields INVALID_PTX); the MMA emits
PTX 9.1, which needs a CUDA 13.1+ driver to load (assembles on 13.0, fails at
load with CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Verified bit-exact on sm_120a
(RTX 5050) and sm_121a (GB10).
"""

from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.sys._assembly import inlined_assembly
from std.sys import _RegisterPackType
from std.memory import UnsafePointer
from layout import LayoutTensor


@always_inline
def sf_rank5_off(row: Int, blk: Int, g_k: Int) -> Int:
    """Byte offset of the scale for output `row`, block `blk` (=k//16) in MAX's
    rank-5 SF tensor (row-major `[ceil(rows/128), g_k, 32, 4, 4]`, g_k=ceil(K/64)).
    """
    return (
        (((row // 128) * g_k + blk // 4) * 32 + (row % 32)) * 4
        + ((row % 128) // 32)
    ) * 4 + (blk % 4)


def _nvfp4_tc_kernel(
    c_out: UnsafePointer[Float32, MutAnyOrigin],
    a_bytes: UnsafePointer[UInt8, ImmutAnyOrigin],
    b_bytes: UnsafePointer[UInt8, ImmutAnyOrigin],
    a_sc: UnsafePointer[UInt8, ImmutAnyOrigin],
    b_sc: UnsafePointer[UInt8, ImmutAnyOrigin],
    n: Int,
    k: Int,
):
    var kb = k // 2
    var g_k = (k + 63) // 64
    var wm = Int(block_idx.y)
    var wn = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var gid = lane >> 2
    var tid = lane & 3
    var mrow = wm * 16
    var ncol = wn * 8

    @parameter
    def lda(
        p: UnsafePointer[UInt8, ImmutAnyOrigin], row: Int, kk: Int
    ) -> UInt32:
        return (p + (row * kb + (kk >> 1))).bitcast[UInt32]()[0]

    @parameter
    def lds5(
        p: UnsafePointer[UInt8, ImmutAnyOrigin], row: Int, blk: Int
    ) -> UInt32:
        return (p + sf_rank5_off(row, blk, g_k)).bitcast[UInt32]()[0]

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    for k0 in range(0, k, 64):
        var a0 = lda(a_bytes, mrow + gid, k0 + tid * 8)
        var a1 = lda(a_bytes, mrow + gid + 8, k0 + tid * 8)
        var a2 = lda(a_bytes, mrow + gid, k0 + tid * 8 + 32)
        var a3 = lda(a_bytes, mrow + gid + 8, k0 + tid * 8 + 32)
        var b0 = lda(b_bytes, ncol + gid, k0 + tid * 8)
        var b1 = lda(b_bytes, ncol + gid, k0 + tid * 8 + 32)

        var blk = k0 // 16
        var sca = UInt32(0)
        if tid == 0:
            sca = lds5(a_sc, mrow + gid, blk)
        elif tid == 1:
            sca = lds5(a_sc, mrow + gid + 8, blk)
        var scb = UInt32(0)
        if tid == 0:
            scb = lds5(b_sc, ncol + gid, blk)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
                ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3},"
                " {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0, 0},"
                " {$15}, {0, 0};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sca, scb)
        c0 = r[0]
        c1 = r[1]
        c2 = r[2]
        c3 = r[3]

    c_out[(mrow + gid) * n + ncol + 2 * tid] = c0
    c_out[(mrow + gid) * n + ncol + 2 * tid + 1] = c1
    c_out[(mrow + gid + 8) * n + ncol + 2 * tid] = c2
    c_out[(mrow + gid + 8) * n + ncol + 2 * tid + 1] = c3


def nvfp4_tensorcore_block_scaled_matmul(
    c: LayoutTensor[mut=True, DType.float32, ...],
    a: LayoutTensor[mut=False, DType.uint8, ...],
    b: LayoutTensor[mut=False, DType.uint8, ...],
    a_scales: LayoutTensor[mut=False, DType.float8_e4m3fn, ...],
    b_scales: LayoutTensor[mut=False, DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """NVFP4 GEMM on consumer-Blackwell tensor cores. `a`/`b` are FP4-packed
    row-major `[M, K/2]` / `[N, K/2]`; `a_scales`/`b_scales` are the rank-5
    e4m3 SF tensors; `c` is `[M, N]` float32. Requires M%16==0, N%8==0, K%64==0.
    """
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]()) * 2
    ctx.enqueue_function[_nvfp4_tc_kernel](
        c.ptr.bitcast[Float32](),
        a.ptr.bitcast[UInt8](),
        b.ptr.bitcast[UInt8](),
        a_scales.ptr.bitcast[UInt8](),
        b_scales.ptr.bitcast[UInt8](),
        N,
        K,
        grid_dim=(N // 8, M // 16),
        block_dim=32,
    )
