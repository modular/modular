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
"""8x8 `simdgroup_matrix` GEMM kernel for Apple Silicon GPUs (M1-M4)."""

from max.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from max.gpu.host import DeviceAttribute, DeviceContext
from layout import Idx, TensorLayout, TensorEngine, TileTensor
from layout.coord import Coord
from std.math import ceildiv
from std.sys import size_of
from std.utils import Index, IndexList
from std.utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type


# ===----------------------------------------------------------------------=== #
# M1-M4: 8x8 simdgroup_matrix MMA (uses stdlib `_mma_apple_8x8`)
# ===----------------------------------------------------------------------=== #

comptime MMA8_DIM = 8
comptime FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane


@inline(.always)
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout (ground-truthed via Metal
    `thread_elements()`). Lane owns (row, col_base) and (row, col_base+1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@inline(.always)
def _mma8_k_block[
    transpose_b: Bool,
    NT_M: Int,
    NT_N: Int,
    KW: Int,
    INTERIOR: Bool,
](
    mut accum: Array[SIMD[.float32, FRAG8], NT_M * NT_N],
    a_slab: TileTensor,
    b_slab: TileTensor,
    m: Int,
    n: Int,
    kb: Int,
    row_base: Int,
    col_base: Int,
    frow: Int,
    fcol: Int,
):
    """Accumulates one `8 * KW`-deep K block into the simdgroup's subtile.

    K is permuted within the block (index `j` at substep `s` reads
    `kb + j * KW + s`) so each lane's fragment is one contiguous aligned load;
    the alignment relies on dispatch gating `k % 16 == 0`.
    """
    comptime a_type = a_slab.dtype
    comptime b_type = b_slab.dtype
    comptime A_W = 2 * KW
    var a_runs = Array[SIMD[a_type, A_W], NT_M](uninitialized=True)
    comptime for mi in range(NT_M):
        var row = mi * MMA8_DIM + frow
        if INTERIOR or row_base + row < m:
            a_runs[mi] = a_slab.load_linear[
                width=A_W, alignment=A_W * size_of[a_type]()
            ](IndexList[2](row, kb + fcol * KW))
        else:
            a_runs[mi] = SIMD[a_type, A_W](0)

    var b_runs = Array[SIMD[b_type, KW], 2 * NT_N](uninitialized=True)
    comptime if transpose_b:
        comptime for ni in range(NT_N):
            comptime for e in range(FRAG8):
                var col = ni * MMA8_DIM + fcol + e
                if INTERIOR or col_base + col < n:
                    b_runs[2 * ni + e] = b_slab.load_linear[
                        width=KW, alignment=KW * size_of[b_type]()
                    ](IndexList[2](col, kb + frow * KW))
                else:
                    b_runs[2 * ni + e] = SIMD[b_type, KW](0)

    comptime for s in range(KW):
        var afrag = Array[SIMD[a_type, FRAG8], NT_M](uninitialized=True)
        comptime for mi in range(NT_M):
            afrag[mi] = SIMD[a_type, FRAG8](a_runs[mi][s], a_runs[mi][KW + s])
        var bfrag = Array[SIMD[b_type, FRAG8], NT_N](uninitialized=True)
        comptime for ni in range(NT_N):
            comptime if transpose_b:
                bfrag[ni] = SIMD[b_type, FRAG8](
                    b_runs[2 * ni][s], b_runs[2 * ni + 1][s]
                )
            else:
                # Odd `n` allows only element alignment.
                var krow = kb + frow * KW + s
                var col = ni * MMA8_DIM + fcol
                if INTERIOR or col_base + col + 1 < n:
                    bfrag[ni] = b_slab.load_linear[
                        width=FRAG8, alignment=size_of[b_type]()
                    ](IndexList[2](krow, col))
                else:
                    var bf = SIMD[b_type, FRAG8](0)
                    if col_base + col < n:
                        bf[0] = b_slab.load_linear[width=1](
                            IndexList[2](krow, col)
                        )
                    bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                var c_frag = accum[mi * NT_N + ni]
                _mma_apple_8x8(
                    accum[mi * NT_N + ni], afrag[mi], bfrag[ni], c_frag
                )


@inline(.always)
def _mma8_k_loop[
    transpose_b: Bool,
    SG_M: Int,
    SG_N: Int,
    INTERIOR: Bool,
](
    mut accum: Array[
        SIMD[.float32, FRAG8], (SG_M // MMA8_DIM) * (SG_N // MMA8_DIM)
    ],
    a: TileTensor,
    b: TileTensor,
    m: Int,
    n: Int,
    k: Int,
    row_base: Int,
    col_base: Int,
    frow: Int,
    fcol: Int,
):
    """Runs the full K reduction.

    Slabs are tiled outside the K loop because AIR does not hoist a `.tile`
    base. NN keeps the natural K order: its B is strided in K.
    """
    comptime NT_M = SG_M // MMA8_DIM
    comptime NT_N = SG_N // MMA8_DIM
    var a_slab = a.tile(Coord(Idx[SG_M], k), Coord(row_base // SG_M, 0))
    comptime if transpose_b:
        var b_slab = b.tile(Coord(Idx[SG_N], k), Coord(col_base // SG_N, 0))
        var k_main = k - k % (MMA8_DIM * 4)
        for kb in range(0, k_main, MMA8_DIM * 4):
            _mma8_k_block[transpose_b, NT_M, NT_N, 4, INTERIOR](
                accum, a_slab, b_slab, m, n, kb, row_base, col_base, frow, fcol
            )
        if k_main < k:
            _mma8_k_block[transpose_b, NT_M, NT_N, 2, INTERIOR](
                accum,
                a_slab,
                b_slab,
                m,
                n,
                k_main,
                row_base,
                col_base,
                frow,
                fcol,
            )
    else:
        var b_slab = b.tile(Coord(k, Idx[SG_N]), Coord(0, col_base // SG_N))
        for kb in range(0, k, MMA8_DIM):
            _mma8_k_block[transpose_b, NT_M, NT_N, 1, INTERIOR](
                accum, a_slab, b_slab, m, n, kb, row_base, col_base, frow, fcol
            )


@inline(.always)
def _mma8_store[
    c_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    NT_M: Int,
    NT_N: Int,
    INTERIOR: Bool,
](
    c: TileTensor[mut=True, c_type, ...],
    accum: Array[SIMD[.float32, FRAG8], NT_M * NT_N],
    m: Int,
    n: Int,
    row_base: Int,
    col_base: Int,
    frow: Int,
    fcol: Int,
):
    """Stores the simdgroup's subtile, guarding each element only on edges."""
    comptime for mi in range(NT_M):
        comptime for ni in range(NT_N):
            var frag = accum[mi * NT_N + ni].cast[c_type]()
            var row = row_base + mi * MMA8_DIM + frow
            var col = col_base + ni * MMA8_DIM + fcol
            comptime if INTERIOR:
                # Odd `n` allows only element alignment.
                comptime if elementwise_lambda_fn:
                    comptime ep = elementwise_lambda_fn.value()
                    ep[c_type, FRAG8, alignment=size_of[c_type]()](
                        Index(row, col), frag
                    )
                else:
                    c.store_linear[alignment=size_of[c_type]()](
                        IndexList[2](row, col), frag
                    )
            else:
                comptime for s in range(FRAG8):
                    if row < m and col + s < n:
                        comptime if elementwise_lambda_fn:
                            comptime ep = elementwise_lambda_fn.value()
                            ep[c_type, 1](Index(row, col + s), frag[s])
                        else:
                            c.store_linear[alignment=size_of[c_type]()](
                                IndexList[2](row, col + s),
                                SIMD[c_type, 1](frag[s]),
                            )


@inline(.always)
def _simdgroup8x8_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    NUM_SIMDGROUPS: Int,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=_],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=_],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=_],
    m: Int,
    n: Int,
    k: Int,
):
    """8x8 simdgroup-matrix GEMM for M1-M4 (no neural accelerator).

    BLOCK_M x BLOCK_N block, 4 simdgroups in a 2x2 grid, each owning a
    (BLOCK_M/2) x (BLOCK_N/2) subtile of 8x8 simdgroup-matrix tiles (f32
    accumulators). `enqueue_apple_matmul_8x8` picks the block per shape.
    Fully-interior simdgroup subtiles take an unguarded fast path; edge subtiles
    zero-fill OOB rows/cols so ragged M/N is correct (in-bounds outputs only use
    in-bounds A rows / B cols; K is always full since dispatch gates k%16==0).
    f32 accum, cast to `c_type` + optional epilogue.
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"
    comptime assert NUM_SIMDGROUPS == 4, "8x8 path assumes 4 simdgroups"

    comptime SG_M = BLOCK_M // 2
    comptime SG_N = BLOCK_N // 2
    comptime NT_M = SG_M // MMA8_DIM
    comptime NT_N = SG_N // MMA8_DIM

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = block_idx.y * BLOCK_M + (sg // 2) * SG_M
    var col_base = block_idx.x * BLOCK_N + (sg % 2) * SG_N
    # Fully-interior simdgroup subtile -> unguarded loads and stores.
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    var accum = Array[SIMD[.float32, FRAG8], NT_M * NT_N](
        fill=SIMD[.float32, FRAG8](0)
    )

    if interior:
        _mma8_k_loop[transpose_b, SG_M, SG_N, True](
            accum, a, b, m, n, k, row_base, col_base, frow, fcol
        )
        _mma8_store[c_type, elementwise_lambda_fn, NT_M, NT_N, True](
            c, accum, m, n, row_base, col_base, frow, fcol
        )
    else:
        _mma8_k_loop[transpose_b, SG_M, SG_N, False](
            accum, a, b, m, n, k, row_base, col_base, frow, fcol
        )
        _mma8_store[c_type, elementwise_lambda_fn, NT_M, NT_N, False](
            c, accum, m, n, row_base, col_base, frow, fcol
        )


@__name(
    t"gemm_kernel_apple_8x8_{c_type}_{a_type}_{b_type}_{transpose_b}"
    t"_{BLOCK_M}x{BLOCK_N}"
)
def gemm_kernel_apple_8x8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    BLOCK_M: Int = 64,
    BLOCK_N: Int = 64,
    BLOCK_K: Int = 16,
    NUM_SIMDGROUPS: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int32,
    n: Int32,
    k: Int32,
):
    """Launchable wrapper for the 8x8 simdgroup-matrix GEMM (bench/test)."""
    var _m = Int(m)
    var _n = Int(n)
    var _k = Int(k)
    _simdgroup8x8_matmul_kernel[
        c_type,
        a_type,
        b_type,
        c_layout,
        a_layout,
        b_layout,
        transpose_b,
        elementwise_lambda_fn,
        s_type,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_SIMDGROUPS,
    ](c, a, b, _m, _n, _k)


# TODO(MOCO-5006): thresholds were measured on a 40-core M5 Max; core-count
# scaling is unverified on M1-M4. Re-sweep there.
comptime _BLOCK_CALIBRATION_CORES = 40


def _apple_8x8_block_choice(m: Int, n: Int, cores: Int) -> Int:
    """Picks the largest block whose grid still fills the GPU: 0 = 16x32,
    1 = 32x32, 2 = 64x32, 3 = 64x64 (BLOCK_M x BLOCK_N)."""
    var elems = m * n * _BLOCK_CALIBRATION_CORES // max(cores, 1)
    if elems <= 64 * 1024:
        return 0
    if elems < 1024 * 1024:
        return 1
    if elems < 4 * 1024 * 1024:
        return 2
    return 3


def enqueue_apple_matmul_8x8[
    *,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Launches the 8x8 GEMM with a block chosen for the output shape.
    Requires `k % 16 == 0`."""
    var cores: Int
    try:
        cores = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        cores = _BLOCK_CALIBRATION_CORES
    var choice = _apple_8x8_block_choice(m, n, cores)
    comptime for i in range(4):
        if choice == i:
            comptime BM = 16 if i == 0 else (32 if i == 1 else 64)
            comptime BN = 64 if i == 3 else 32
            comptime kernel = gemm_kernel_apple_8x8[
                c.dtype,
                a.dtype,
                b.dtype,
                type_of(c).LayoutType,
                type_of(a).LayoutType,
                type_of(b).LayoutType,
                type_of(c).Engine,
                type_of(a).Engine,
                type_of(b).Engine,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=16,
                NUM_SIMDGROUPS=4,
            ]
            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(4 * WARP_SIZE,),
            )
