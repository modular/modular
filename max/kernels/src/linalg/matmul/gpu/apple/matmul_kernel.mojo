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
"""Simdgroup-tiled Apple M5 matmul kernel built on MmaOpApple.

64x64 output tile per threadgroup; four simdgroups (128 threads) each own a
32x32 subtile (2x2 MmaOpApple). A per-simdgroup runtime branch picks between
an unbounded fast path and a bounded path that handles ragged M/N edges and
partial K tails.

Block-to-tile: each threadgroup decodes `block_idx.x` via
`morton_decode_2d_rect` over a `side_m * side_n` grid (each axis padded
to the next pow2). Threadgroups outside `(grid_m, grid_n)` early-return.
"""

from std.collections import Optional
from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.sys import align_of
from std.utils import IndexList
from std.utils.type_functions import ConditionalType
from layout import TileTensor, Idx
from layout.tile_layout import Layout, TensorLayout, row_major
from layout.coord import Coord
from linalg.arch.apple.mma import MmaOpApple
from linalg.utils import elementwise_epilogue_type


comptime BM: Int = 64
comptime BN: Int = 64
comptime BK: Int = 16
comptime SG_M: Int = 32  # simdgroup subtile rows (2 * MMA_M)
comptime SG_N: Int = 32  # simdgroup subtile cols (2 * MMA_N)
comptime NUM_SG_M: Int = BM / SG_M
comptime NUM_SG_N: Int = BN / SG_N
comptime NUM_SG: Int = NUM_SG_M * NUM_SG_N
comptime THREADS_PER_BLOCK: Int = NUM_SG * Int(WARP_SIZE)


@always_inline
def _pick_b_slab_layout[
    transpose_b: Bool
](k: Int, n: Int) -> ConditionalType[
    Trait=TensorLayout,
    If=transpose_b,
    Then=type_of(
        Layout(
            Coord(k, Idx[SG_N]),
            Coord(Idx[1], k),
        )
    ),
    Else=type_of(
        Layout(
            Coord(k, Idx[SG_N]),
            Coord(n, Idx[1]),
        )
    ),
]:
    """B-slab Layout selected at comptime.

    `transpose_b=True`  -> strides `(1, K)` (col_major view of an (N,K) buffer).
    `transpose_b=False` -> strides `(N, 1)` (row_major, inherits parent N stride).
    """
    comptime _Ret = ConditionalType[
        Trait=TensorLayout,
        If=transpose_b,
        Then=type_of(
            Layout(
                Coord(k, Idx[SG_N]),
                Coord(Idx[1], k),
            )
        ),
        Else=type_of(
            Layout(
                Coord(k, Idx[SG_N]),
                Coord(n, Idx[1]),
            )
        ),
    ]
    comptime if transpose_b:
        return rebind[_Ret](
            Layout(
                Coord(k, Idx[SG_N]),
                Coord(Idx[1], k),
            )
        )
    else:
        return rebind[_Ret](
            Layout(
                Coord(k, Idx[SG_N]),
                Coord(n, Idx[1]),
            )
        )


@always_inline
def morton_decode_2d(flat_idx: UInt32) -> Tuple[UInt32, UInt32]:
    """Decode a linear index to (tile_m, tile_n) via Morton Z-order.

    Even bits of flat_idx -> tile_n, odd bits -> tile_m. The decoded pair
    may fall outside any rectangular grid that isn't a power-of-2 square;
    the caller checks bounds.
    """
    var x = flat_idx & 0x55555555
    var y = (flat_idx >> 1) & 0x55555555

    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF

    y = (y | (y >> 1)) & 0x33333333
    y = (y | (y >> 2)) & 0x0F0F0F0F
    y = (y | (y >> 4)) & 0x00FF00FF
    y = (y | (y >> 8)) & 0x0000FFFF

    return (y, x)


@always_inline
def morton_decode_2d_rect(
    flat_idx: UInt32,
    log2_m: UInt32,
    log2_n: UInt32,
) -> Tuple[UInt32, UInt32]:
    """Decode `flat_idx` to (tile_m, tile_n) over a `(1<<log2_m) x (1<<log2_n)` grid.

    Z-order covers a `min(side_m, side_n)` square core; remaining bits sweep
    the longer axis. Reduces to `morton_decode_2d` when `log2_m == log2_n`.
    """
    var log2_lo = min(log2_m, log2_n)
    var lo_mask = (UInt32(1) << (UInt32(2) * log2_lo)) - UInt32(1)
    var lo_mn = morton_decode_2d(flat_idx & lo_mask)
    var hi_bits = flat_idx >> (UInt32(2) * log2_lo)

    var m_extra: UInt32 = (hi_bits << log2_lo) if log2_m > log2_n else UInt32(0)
    var n_extra: UInt32 = (hi_bits << log2_lo) if log2_n > log2_m else UInt32(0)

    return (lo_mn[0] | m_extra, lo_mn[1] | n_extra)


def apple_matmul_kernel[
    in_type: DType,
    c_type: DType = DType.float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    d_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[in_type], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[in_type], MutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    log2_grid_m: UInt32,
    log2_grid_n: UInt32,
):
    """Apple M5 simdgroup-tiled GEMM: one 64x64 tile per threadgroup.

    Grid is `(1<<log2_grid_m) * (1<<log2_grid_n)` threadgroups of 128 threads;
    OOB threadgroups early-return after Morton decode. For `transpose_b=True`,
    B is the `(N, K)` row-major buffer reinterpreted as `col_major(K, N)`.
    """
    # Apple's scalar ALU is faster on 32-bit math; use Int32 locally and cast
    # back to Int only at API boundaries (.tile[], MmaOpApple).
    var m_i32 = Int32(m)
    var n_i32 = Int32(n)
    var k_i32 = Int32(k)
    comptime BM_i32: Int32 = Int32(BM)
    comptime BN_i32: Int32 = Int32(BN)
    comptime BK_i32: Int32 = Int32(BK)
    comptime SG_M_i32: Int32 = Int32(SG_M)
    comptime SG_N_i32: Int32 = Int32(SG_N)
    comptime NUM_SG_N_i32: Int32 = Int32(NUM_SG_N)

    var grid_m = (m_i32 + BM_i32 - 1) // BM_i32
    var grid_n = (n_i32 + BN_i32 - 1) // BN_i32

    var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
    var sg_m_idx = sg_id // NUM_SG_N_i32
    var sg_n_idx = sg_id % NUM_SG_N_i32

    var flat_idx = UInt32(block_idx.x)

    var tile_mn = morton_decode_2d_rect(flat_idx, log2_grid_m, log2_grid_n)
    var tile_m = Int32(tile_mn[0])
    var tile_n = Int32(tile_mn[1])
    if tile_m >= grid_m or tile_n >= grid_n:
        return

    comptime Mma = MmaOpApple[
        DType.float32,
        in_type,
        num_m_mmas=2,
        num_n_mmas=2,
    ]
    var mma_op = Mma()
    var accum = Mma.zero_accum()

    var row_base = tile_m * BM_i32 + sg_m_idx * SG_M_i32
    var col_base = tile_n * BN_i32 + sg_n_idx * SG_N_i32
    var sg_row_idx = row_base // SG_M_i32
    var sg_col_idx = col_base // SG_N_i32

    var sg_m_end = row_base + SG_M_i32
    var sg_n_end = col_base + SG_N_i32
    var m_n_edge = (sg_m_end > m_i32) or (sg_n_end > n_i32)

    # Skip fully-OOB simdgroups: there's no later
    # threadgroup-uniform op, so the early return is safe.
    if row_base >= m_i32 or col_base >= n_i32:
        return

    var k_full_strips = k_i32 // BK_i32
    var has_k_tail = (k_i32 % BK_i32) != 0

    # A slab: shape (SG_M, k), strides (k, 1). Tiling in the K-loop uses
    # (0, k16), making the row offset loop-invariant.
    var a_row_offset = Int(sg_row_idx * SG_M_i32 * k_i32)
    var a_slab = TileTensor(
        a_ptr + a_row_offset,
        Layout(
            Coord(Idx[SG_M], k),
            Coord(k, Idx[1]),
        ),
    )

    # B slab: col_major(K, N) for transpose_b (stride[0]=1, stride[1]=K)
    # or row_major(K, N) (stride[0]=N, stride[1]=1) over the parent buffer.
    var b_col_offset = Int(
        sg_col_idx * SG_N_i32 * k_i32
    ) if transpose_b else Int(sg_col_idx * SG_N_i32)
    var b_slab = TileTensor(
        b_ptr + b_col_offset, _pick_b_slab_layout[transpose_b](k, n)
    )

    # Stay on the fp32-out, no-lambda fast path for parity with PR #84575;
    # everything else flows through the per-fragment epilogue below.
    comptime use_epilogue_path = (
        c_type != DType.float32 or elementwise_lambda_fn
    )

    # Cast-then-store. Matches AMD's contract: the lambda receives
    # `SIMD[c_type, width]`. See `AMDMatmul.run`'s `elementwise_lambda_fn`
    # branch in `amd_matmul.mojo`.
    @always_inline
    @parameter
    def _apply_epilogue[bounded: Bool](tile_row_base: Int, tile_col_base: Int):
        @always_inline
        @parameter
        def _write_one(row: Int, col: Int, x_fp32: SIMD[DType.float32, 1]):
            var y_ctype = x_fp32.cast[c_type]()
            comptime if elementwise_lambda_fn:
                comptime epilogue = elementwise_lambda_fn.value()
                epilogue[c_type, 1](IndexList[2](row, col), y_ctype)
            else:
                d_ptr[row * n + col] = y_ctype[0]

        @always_inline
        @parameter
        def _write_four(row: Int, col0: Int, x_fp32: SIMD[DType.float32, 4]):
            var y_ctype = x_fp32.cast[c_type]()
            comptime if elementwise_lambda_fn:
                comptime epilogue = elementwise_lambda_fn.value()
                # Element alignment only: `row * n` is unaligned for odd `n`.
                comptime align = align_of[Scalar[c_type]]()
                epilogue[c_type, 4, alignment=align](
                    IndexList[2](row, col0), y_ctype
                )
            else:
                (d_ptr + row * n + col0).store(y_ctype)

        @always_inline
        @parameter
        def _write_half(row: Int, col0: Int, v_fp32: SIMD[DType.float32, 4]):
            comptime if bounded:
                if row < m:
                    if col0 + 3 < n:
                        _write_four(row, col0, v_fp32)
                    else:
                        var amt = min(4, n - col0)
                        for e in range(amt):
                            _write_one(
                                row,
                                col0 + e,
                                SIMD[DType.float32, 1](v_fp32[e]),
                            )
            else:
                _write_four(row, col0, v_fp32)

        comptime for mi in range(Mma.num_m_mmas):
            comptime for ni in range(Mma.num_n_mmas):
                var frag_fp32 = accum[mi * Mma.num_n_mmas + ni]
                var col0 = tile_col_base + ni * 16 + Int(mma_op.cb)
                var row_lo = tile_row_base + mi * 16 + Int(mma_op.rb)
                var row_hi = row_lo + 8
                _write_half(row_lo, col0, frag_fp32.slice[4, offset=0]())
                _write_half(row_hi, col0, frag_fp32.slice[4, offset=4]())

    # `rebind` to fp32 so `mma_op.store{,_bounded}` typechecks; this branch
    # is only entered when `c_type == fp32` (use_epilogue_path is False),
    # so the rebind is a no-op at runtime.
    @always_inline
    @parameter
    def _fast_path_store[
        bounded: Bool
    ](valid_rows: Int = 0, valid_cols: Int = 0):
        var d_ptr_fp32 = rebind[
            UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
        ](d_ptr)
        var d_mat_fp32 = TileTensor(d_ptr_fp32, row_major(m, n))
        var d_sub_fp32 = d_mat_fp32.tile[SG_M, SG_N](
            Int(sg_row_idx), Int(sg_col_idx)
        )
        comptime if bounded:
            mma_op.store_bounded(accum, d_sub_fp32, valid_rows, valid_cols)
        else:
            mma_op.store(accum, d_sub_fp32)

    # K-loop loads A/B directly from device memory each step.
    # Threadgroup-memory staging *degrades* matmul on Apple Silicon.
    if m_n_edge:
        # Belt-and-suspenders clamp. The early return guarantees
        # `[1, SG_M]` in steady state; this survives a future refactor
        # that drops it. Inside `MmaOpApple.mma`, `valid_* - mi*16`
        # may still go negative for partial tiles -- `_bounded_load`
        # zero-fills, which is correct.
        var valid_rows = max(Int32(1), min(SG_M_i32, m_i32 - row_base))
        var valid_cols = max(Int32(1), min(SG_N_i32, n_i32 - col_base))
        var tail_count: Int32 = 1 if has_k_tail else 0
        var k_total_strips = k_full_strips + tail_count
        for k16 in range(k_total_strips):
            var k_valid = min(BK_i32, k_i32 - k16 * BK_i32)
            var a_sub = a_slab.tile[SG_M, BK](0, Int(k16))
            var b_sub = b_slab.tile[BK, SG_N](Int(k16), 0)
            mma_op.mma[bounded=True](
                accum,
                a_sub,
                b_sub,
                a_valid_rows=Int(valid_rows),
                b_valid_cols=Int(valid_cols),
                k_valid=Int(k_valid),
            )
        comptime if use_epilogue_path:
            _apply_epilogue[bounded=True](Int(row_base), Int(col_base))
        else:
            _fast_path_store[bounded=True](Int(valid_rows), Int(valid_cols))
    else:
        for k16 in range(k_full_strips):
            var a_sub = a_slab.tile[SG_M, BK](0, Int(k16))
            var b_sub = b_slab.tile[BK, SG_N](Int(k16), 0)
            mma_op.mma(accum, a_sub, b_sub)
        if has_k_tail:
            var k_tail = k_i32 - k_full_strips * BK_i32
            var a_sub = a_slab.tile[SG_M, BK](0, Int(k_full_strips))
            var b_sub = b_slab.tile[BK, SG_N](Int(k_full_strips), 0)
            mma_op.mma[bounded=True](
                accum,
                a_sub,
                b_sub,
                a_valid_rows=SG_M,
                b_valid_cols=SG_N,
                k_valid=Int(k_tail),
            )
        comptime if use_epilogue_path:
            _apply_epilogue[bounded=False](Int(row_base), Int(col_base))
        else:
            _fast_path_store[bounded=False]()


@always_inline
def enqueue_apple_matmul[
    in_type: DType,
    c_type: DType = DType.float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    b: TileTensor[in_type, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the Apple M5 matmul kernel on the given device context.

    Accepts row-major TileTensor operands. For `transpose_b=True`, B is
    expected with shape `(N, K)`.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
        M1-M4 lack GPU `neural accelerator`; future generations require
        re-validation.
    """
    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_apple_matmul requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
            (
                ". Route M1-M4 to the naive matmul path; re-validate for"
                " future generations."
            ),
        )

    comptime assert (
        c_type == DType.float16
        or c_type == DType.bfloat16
        or c_type == DType.float32
    ), "enqueue_apple_matmul: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(
        Int(c.dim[0]()) == m and Int(c.dim[1]()) == n, "C shape (M, N)"
    )
    comptime if transpose_b:
        debug_assert(
            Int(b.dim[0]()) == n,
            "transpose_b=True expects B shape (N, K)",
        )
        debug_assert(
            Int(b.dim[1]()) == k,
            "transpose_b=True expects B shape (N, K)",
        )
    else:
        debug_assert(
            Int(b.dim[0]()) == k,
            "transpose_b=False expects B shape (K, N)",
        )
        debug_assert(
            Int(b.dim[1]()) == n,
            "transpose_b=False expects B shape (K, N)",
        )

    # MmaOpApple narrows the row stride to UInt16 (see `_load_fragment` /
    # `_store_fragment` in linalg/arch/apple/mma.mojo). Catch the wrap here:
    # NN A-slab stride = K, NN B-slab stride = N; NT B-slab stride = K (covered).
    debug_assert(k <= 65535, "Apple matmul: K must fit in UInt16; got K=", k)
    comptime if not transpose_b:
        debug_assert(
            n <= 65535,
            "Apple matmul (NN): N must fit in UInt16; got N=",
            n,
        )

    # Per-axis next-pow2 grid for rectangular Z-order. e.g. 32x224 (Llama-3
    # MLP up-proj) -> 32x256 = 8192 launches vs the prior square 256x256 = 65536.
    var grid_m = (m + BM - 1) // BM
    var grid_n = (n + BN - 1) // BN

    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    var grid_dim = side_m * side_n

    comptime kernel = apple_matmul_kernel[
        in_type=in_type,
        c_type=c_type,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    ctx.enqueue_function[kernel](
        c.ptr,
        a.ptr,
        b.ptr,
        m,
        n,
        k,
        log2_m,
        log2_n,
        grid_dim=(grid_dim),
        block_dim=(THREADS_PER_BLOCK),
    )
