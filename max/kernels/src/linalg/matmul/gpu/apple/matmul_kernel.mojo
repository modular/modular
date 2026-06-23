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
"""Structured Apple M5 simdgroup-tiled matmul (Metal 4 hardware MMA).

Everything lives in the `AppleM5MatMul` struct (mirroring the AMD/NVIDIA
structured kernels): the comptime config, the Morton tile scheduler, the
B-layout helper, the single-pass GPU kernel (`run`), and the split-K kernels
(`run_split_k_partial` / `run_split_k_reduce`). The `enqueue_apple_matmul` /
`enqueue_apple_matmul_split_k` free functions are the host-side launchers
(kept standalone so callers and tests dispatch without naming the struct).

64x64 output tile per threadgroup; four simdgroups (128 threads) each own a
32x32 subtile (2x2 `MmaOpApple`). A per-simdgroup runtime branch picks between
an unbounded fast path and a bounded path for ragged M/N edges and partial K
tails. Operands load DRAM->register directly -- threadgroup-memory staging
*degrades* matmul on Apple Silicon. See `kernels/apple-m5-matmul` in the KB.
"""

from std.collections import Optional
from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.sys import align_of
from std.utils import IndexList
from std.utils.type_functions import ConditionalType
from layout import TileTensor, Idx
from layout.tile_layout import Layout, TensorLayout, row_major
from layout.coord import Coord
from linalg.arch.apple.mma import MmaOpApple
from linalg.utils import elementwise_epilogue_type


struct AppleM5MatMul[
    in_type: DType,
    c_type: DType = DType.float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Apple M5 simdgroup-tiled GEMM (Metal 4 hardware MMA).

    Parameters:
        in_type: A/B element type (fp16, bf16, fp32; int8 at the MMA-op level).
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        transpose_b: If True, B is `(N, K)` row-major (viewed as `col_major(K, N)`);
            otherwise B is `(K, N)` row-major.
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract).

    `run` is the GPU kernel entry (TileTensor operands; `M`/`N`/`K` derived from
    them). `run_split_k_partial` / `run_split_k_reduce` are the split-K kernels.
    Launch via `enqueue_apple_matmul` / `enqueue_apple_matmul_split_k`.
    """

    # === Comptime tile config. 64x64 block, 32x32 simdgroup, BK=16 K-strip. ===
    comptime BM = 64
    comptime BN = 64
    comptime BK = 16
    comptime SG_M = 32  # simdgroup subtile rows (2 * MMA_M)
    comptime SG_N = 32  # simdgroup subtile cols (2 * MMA_N)
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * Int(WARP_SIZE)
    comptime REDUCE_BLOCK = 256  # threads/block for the split-K reduce kernel
    # 2x2 (4 accumulators/simdgroup) is the benchmarked optimum on M5 Max;
    # 2x4 / 4x2 regress 26-60% -- the MMA pipeline is not latency-starved at
    # 4 accumulators, so more only adds register pressure / spills.
    comptime Mma = MmaOpApple[
        DType.float32, Self.in_type, num_m_mmas=2, num_n_mmas=2
    ]

    # === Morton (Z-order) tile scheduling ================================== #

    @staticmethod
    def morton_decode_2d(flat_idx: UInt32) -> Tuple[UInt32, UInt32]:
        """Decode a linear index to (tile_m, tile_n) via Morton Z-order.

        Even bits of flat_idx -> tile_n, odd bits -> tile_m. The decoded pair
        may fall outside any rectangular grid that isn't a power-of-2 square;
        the caller checks bounds.

        Future: as of AIR 4.1 a `bit_interleave` intrinsic can do this
        interleave directly, replacing the hand-rolled bit-scatter below. A
        Gilbert-curve dispatch order was also explored for better locality,
        but it needs a per-shape predispatch kernel to generate the order.
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

    @staticmethod
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
        var lo_mn = Self.morton_decode_2d(flat_idx & lo_mask)
        var hi_bits = flat_idx >> (UInt32(2) * log2_lo)

        var m_extra: UInt32 = (
            hi_bits << log2_lo
        ) if log2_m > log2_n else UInt32(0)
        var n_extra: UInt32 = (
            hi_bits << log2_lo
        ) if log2_n > log2_m else UInt32(0)

        return (lo_mn[0] | m_extra, lo_mn[1] | n_extra)

    # === B-operand layout selection ======================================= #

    @staticmethod
    def _pick_b_mat_layout(
        k: Int, n: Int
    ) -> ConditionalType[
        Trait=TensorLayout,
        If=Self.transpose_b,
        Then=type_of(Layout(Coord(k, n), Coord(Idx[1], k))),
        Else=type_of(Layout(Coord(k, n), Coord(n, Idx[1]))),
    ]:
        """Full B=(K, N) Layout selected at comptime.

        `transpose_b=True`  -> strides `(1, K)` (col_major view of an (N,K) buffer).
        `transpose_b=False` -> strides `(N, 1)` (row_major (K, N)).

        `run` pre-tiles this into a per-simdgroup SG_N-wide column slab
        (full K), then the K-loop tiles only the K axis; no pointer arithmetic.
        """
        comptime _Ret = ConditionalType[
            Trait=TensorLayout,
            If=Self.transpose_b,
            Then=type_of(Layout(Coord(k, n), Coord(Idx[1], k))),
            Else=type_of(Layout(Coord(k, n), Coord(n, Idx[1]))),
        ]
        comptime if Self.transpose_b:
            return rebind[_Ret](Layout(Coord(k, n), Coord(Idx[1], k)))
        else:
            return rebind[_Ret](Layout(Coord(k, n), Coord(n, Idx[1])))

    # === Single-pass kernel ================================================ #

    @__name(
        t"apple_matmul_run_{Self.in_type}_{Self.c_type}_tb{Self.transpose_b}"
    )
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin],
        b: TileTensor[Self.in_type, b_layout, ImmutAnyOrigin],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """GEMM kernel entry; `M`/`N`/`K` derive from the operands.

        A is `(M, K)` row-major and C is `(M, N)` row-major; B is `(K, N)` for
        `transpose_b=False` or `(N, K)` for `transpose_b=True`. Grid is
        `(1<<log2_grid_m) * (1<<log2_grid_n)` threadgroups of 128 threads; OOB
        threadgroups early-return after Morton decode.
        """
        var c_ptr = c.ptr
        var a_ptr = a.ptr
        var b_ptr = b.ptr
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())

        # Apple's scalar ALU is faster on 32-bit math; use Int32 locally and
        # cast back to Int only at API boundaries (.tile[], MmaOpApple).
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N
        comptime Mma = Self.Mma
        # `c_type` / `elementwise_lambda_fn` / `transpose_b` are struct *params*:
        # spelled `Self.x` below (a param can't be aliased to a same-name local
        # the way the members just above are).

        var m_i32 = Int32(m)
        var n_i32 = Int32(n)
        var k_i32 = Int32(k)
        comptime BM_i32: Int32 = Int32(BM)
        comptime BN_i32: Int32 = Int32(BN)
        comptime BK_i32: Int32 = Int32(BK)
        comptime SG_M_i32: Int32 = Int32(SG_M)
        comptime SG_N_i32: Int32 = Int32(SG_N)
        comptime NUM_SG_N_i32: Int32 = Int32(Self.NUM_SG_N)

        var grid_m = (m_i32 + BM_i32 - 1) // BM_i32
        var grid_n = (n_i32 + BN_i32 - 1) // BN_i32

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // NUM_SG_N_i32
        var sg_n_idx = sg_id % NUM_SG_N_i32

        var flat_idx = UInt32(block_idx.x)

        var tile_mn = Self.morton_decode_2d_rect(
            flat_idx, log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var mma_op = Mma()
        var accum = Mma.zero_accum()

        var row_base = tile_m * BM_i32 + sg_m_idx * SG_M_i32
        var col_base = tile_n * BN_i32 + sg_n_idx * SG_N_i32
        var sg_row_idx = row_base // SG_M_i32
        var sg_col_idx = col_base // SG_N_i32

        var sg_m_end = row_base + SG_M_i32
        var sg_n_end = col_base + SG_N_i32
        var is_edge_tile = (sg_m_end > m_i32) or (sg_n_end > n_i32)

        # Skip fully-OOB simdgroups: there's no later threadgroup-uniform op,
        # so the early return is safe.
        if row_base >= m_i32 or col_base >= n_i32:
            return

        var k_full_strips = k_i32 // BK_i32
        var has_k_tail = (k_i32 % BK_i32) != 0

        # Full A=(M, K) row-major and B=(K, N) tensors.
        var a_mat = TileTensor(a_ptr, Layout(Coord(m, k), Coord(k, Idx[1])))
        var b_mat = TileTensor(b_ptr, Self._pick_b_mat_layout(k, n))

        # Pre-tile this simdgroup's SG_M-row block of A and SG_N-col block of B
        # (full K, runtime-`k` shape via the Coord-arg `.tile`). The K-loop then
        # tiles only the K axis (row/col coord 0), so the loop-invariant
        # `sg_row/col * SG * stride` offset is baked into the slab base once and
        # kept out of the hot loop -- recovers the ~3-6% the per-strip
        # `.tile(sg_idx, k16)` form lost to un-hoisted address math (no pointer
        # arithmetic; `.tile` computes the base).
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))
        var b_slab = b_mat.tile(Coord(k, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        # fp32 out with no fused lambda takes the fast `mma_op.store` path;
        # every other (c_type, lambda) combo flows through the epilogue below.
        comptime use_epilogue_path = (
            Self.c_type != DType.float32 or Self.elementwise_lambda_fn
        )

        # Cast-then-store epilogue (non-fp32 out and/or a fused
        # `elementwise_lambda_fn`). Writes through a `.tile`-derived simdgroup
        # view of C -- no pointer arithmetic. The lambda contract matches AMD's:
        # it receives `SIMD[c_type, width]` at absolute (row, col).
        @always_inline
        @parameter
        def _apply_epilogue[
            bounded: Bool
        ](tile_row_base: Int, tile_col_base: Int):
            var c_sub = TileTensor(c_ptr, row_major(m, n)).tile[SG_M, SG_N](
                Int(sg_row_idx), Int(sg_col_idx)
            )
            # 4 contiguous output cols = one SIMD unit. Element alignment only:
            # the row stride `n` is odd for odd N, so the default full-vector
            # alignment would fault -- override it on the vectorized store.
            comptime elem_align = align_of[Scalar[Self.c_type]]()
            var c_vec = c_sub.vectorize[1, 4]()

            @always_inline
            @parameter
            def _write4(
                lrow: Int,
                lcol: Int,
                arow: Int,
                acol: Int,
                v_fp32: SIMD[DType.float32, 4],
            ):
                # `lrow,lcol`: coords inside the simdgroup tile (C store).
                # `arow,acol`: absolute coords (bounds + the lambda contract).
                var y = v_fp32.cast[Self.c_type]()
                comptime if Self.elementwise_lambda_fn:
                    comptime epilogue = Self.elementwise_lambda_fn.value()
                    comptime if bounded:
                        if acol + 3 < n:
                            epilogue[Self.c_type, 4, alignment=elem_align](
                                IndexList[2](arow, acol), y
                            )
                        else:
                            for e in range(min(4, n - acol)):
                                epilogue[Self.c_type, 1](
                                    IndexList[2](arow, acol + e),
                                    SIMD[Self.c_type, 1](y[e]),
                                )
                    else:
                        epilogue[Self.c_type, 4, alignment=elem_align](
                            IndexList[2](arow, acol), y
                        )
                else:
                    comptime if bounded:
                        if acol + 3 < n:
                            c_vec.store[alignment=elem_align](
                                Coord(lrow, lcol // 4), y
                            )
                        else:
                            for e in range(min(4, n - acol)):
                                c_sub.store[width=1, alignment=elem_align](
                                    Coord(lrow, lcol + e),
                                    SIMD[Self.c_type, 1](y[e]),
                                )
                    else:
                        c_vec.store[alignment=elem_align](
                            Coord(lrow, lcol // 4), y
                        )

            comptime for mi in range(Mma.num_m_mmas):
                comptime for ni in range(Mma.num_n_mmas):
                    var frag = accum[mi * Mma.num_n_mmas + ni]
                    var lcol = ni * 16 + Int(mma_op.cb)
                    var lrow = mi * 16 + Int(mma_op.rb)
                    var acol = tile_col_base + lcol
                    var arow = tile_row_base + lrow
                    comptime if bounded:
                        if arow < m:
                            _write4(
                                lrow,
                                lcol,
                                arow,
                                acol,
                                frag.slice[4, offset=0](),
                            )
                        if arow + 8 < m:
                            _write4(
                                lrow + 8,
                                lcol,
                                arow + 8,
                                acol,
                                frag.slice[4, offset=4](),
                            )
                    else:
                        _write4(
                            lrow, lcol, arow, acol, frag.slice[4, offset=0]()
                        )
                        _write4(
                            lrow + 8,
                            lcol,
                            arow + 8,
                            acol,
                            frag.slice[4, offset=4](),
                        )

        # `rebind` to fp32 so `mma_op.store{,_bounded}` typechecks; this branch
        # is only entered when `c_type == fp32` (use_epilogue_path is False),
        # so the rebind is a no-op at runtime.
        @always_inline
        @parameter
        def _fast_path_store[
            bounded: Bool
        ](valid_rows: Int = 0, valid_cols: Int = 0):
            var c_ptr_fp32 = rebind[
                UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
            ](c_ptr)
            var c_mat_fp32 = TileTensor(c_ptr_fp32, row_major(m, n))
            var c_sub_fp32 = c_mat_fp32.tile[SG_M, SG_N](
                Int(sg_row_idx), Int(sg_col_idx)
            )
            comptime if bounded:
                mma_op.store_bounded(accum, c_sub_fp32, valid_rows, valid_cols)
            else:
                mma_op.store(accum, c_sub_fp32)

        # K-loop loads A/B directly from device memory each step.
        # Threadgroup-memory staging *degrades* matmul on Apple Silicon.
        if is_edge_tile:
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

    # === Split-K kernels =================================================== #
    # Partition the K axis across `num_splits` threadgroup-sets so large-K /
    # small-M*N shapes (few output tiles -> low occupancy) get more parallelism.
    # Each split writes an fp32 partial to a workspace; the reduce pass sums the
    # partials and applies the cast + epilogue. Deterministic -- no global
    # atomics (Apple fp32 atomic-add is not relied upon).

    @__name(t"apple_matmul_split_k_partial_{Self.in_type}_tb{Self.transpose_b}")
    @staticmethod
    def run_split_k_partial[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
    ](
        partials_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin],
        b: TileTensor[Self.in_type, b_layout, ImmutAnyOrigin],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
        k_per_split: Int,
    ):
        """One 64x64 output tile's fp32 partial over a BK-aligned K-slice.

        Grid is `(side_m * side_n) * num_splits` threadgroups. The split index
        selects the K-range `[s*k_per_split, min(K, (s+1)*k_per_split))` and the
        `partials[s]` output matrix; the tile index is Morton-decoded as in
        `run`. `k_per_split` is a multiple of `BK`, so every split but the last
        is full BK strips; the last may carry a partial-BK tail. No cast, no
        epilogue -- raw fp32 accumulator out.
        """
        var a_ptr = a.ptr
        var b_ptr = b.ptr
        var m = Int(a.dim[0]())
        var k = Int(a.dim[1]())
        var n = Int(b.dim[0]()) if Self.transpose_b else Int(b.dim[1]())

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N
        comptime Mma = Self.Mma

        var m_i32 = Int32(m)
        var n_i32 = Int32(n)
        var k_i32 = Int32(k)
        comptime BM_i32: Int32 = Int32(BM)
        comptime BN_i32: Int32 = Int32(BN)
        comptime BK_i32: Int32 = Int32(BK)
        comptime SG_M_i32: Int32 = Int32(SG_M)
        comptime SG_N_i32: Int32 = Int32(SG_N)
        comptime NUM_SG_N_i32: Int32 = Int32(Self.NUM_SG_N)

        var grid_m = (m_i32 + BM_i32 - 1) // BM_i32
        var grid_n = (n_i32 + BN_i32 - 1) // BN_i32

        var num_tiles = UInt32(1) << (log2_grid_m + log2_grid_n)
        var bx = UInt32(block_idx.x)
        var split_idx = Int32(bx // num_tiles)
        var tile_flat = bx % num_tiles

        var tile_mn = Self.morton_decode_2d_rect(
            tile_flat, log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var k0 = split_idx * Int32(k_per_split)
        if k0 >= k_i32:
            return
        var k1 = min(k_i32, k0 + Int32(k_per_split))
        var strip0 = k0 // BK_i32
        var span = k1 - k0
        var full_strips = span // BK_i32
        var has_tail = (span % BK_i32) != 0

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // NUM_SG_N_i32
        var sg_n_idx = sg_id % NUM_SG_N_i32

        var mma_op = Mma()
        var accum = Mma.zero_accum()

        var row_base = tile_m * BM_i32 + sg_m_idx * SG_M_i32
        var col_base = tile_n * BN_i32 + sg_n_idx * SG_N_i32
        if row_base >= m_i32 or col_base >= n_i32:
            return
        var sg_row_idx = row_base // SG_M_i32
        var sg_col_idx = col_base // SG_N_i32
        var is_edge_tile = (row_base + SG_M_i32 > m_i32) or (
            col_base + SG_N_i32 > n_i32
        )

        var a_mat = TileTensor(a_ptr, Layout(Coord(m, k), Coord(k, Idx[1])))
        var b_mat = TileTensor(b_ptr, Self._pick_b_mat_layout(k, n))
        # Hoist the simdgroup base offset out of the K-loop (see `run`).
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))
        var b_slab = b_mat.tile(Coord(k, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        var total_strips = Int(full_strips) + (1 if has_tail else 0)
        if is_edge_tile:
            var valid_rows = max(Int32(1), min(SG_M_i32, m_i32 - row_base))
            var valid_cols = max(Int32(1), min(SG_N_i32, n_i32 - col_base))
            for j in range(total_strips):
                var gstrip = strip0 + Int32(j)
                var k_valid = min(BK_i32, k1 - gstrip * BK_i32)
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=Int(valid_rows),
                    b_valid_cols=Int(valid_cols),
                    k_valid=Int(k_valid),
                )
            var part_sub = TileTensor(
                partials_ptr + Int(split_idx) * m * n, row_major(m, n)
            ).tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            mma_op.store_bounded(
                accum, part_sub, Int(valid_rows), Int(valid_cols)
            )
        else:
            for j in range(Int(full_strips)):
                var gstrip = strip0 + Int32(j)
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                mma_op.mma(accum, a_sub, b_sub)
            if has_tail:
                var gstrip = strip0 + full_strips
                var k_valid = k1 - gstrip * BK_i32
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=SG_M,
                    b_valid_cols=SG_N,
                    k_valid=Int(k_valid),
                )
            var part_sub = TileTensor(
                partials_ptr + Int(split_idx) * m * n, row_major(m, n)
            ).tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            mma_op.store(accum, part_sub)

    @__name(t"apple_matmul_split_k_reduce_{Self.c_type}")
    @staticmethod
    def run_split_k_reduce[
        c_layout: TensorLayout,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
        partials_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
        num_splits: Int,
    ):
        """Sum `num_splits` fp32 partials per output element, cast, store / fuse.

        One thread per output element; `idx = block_idx.x * block_dim.x +
        thread_idx.x`. The fused `elementwise_lambda_fn` (if any) sees the
        absolute (row, col) and the final `SIMD[c_type, 1]`.
        """
        # `c_type` / `elementwise_lambda_fn` are struct params -- use `Self.x`.
        var c_ptr = c.ptr
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())

        var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        var total = m * n
        if idx >= total:
            return

        var acc = Float32(0)
        var mn = m * n
        for s in range(num_splits):
            acc += partials_ptr[s * mn + idx]

        var y = acc.cast[Self.c_type]()
        comptime if Self.elementwise_lambda_fn:
            comptime epilogue = Self.elementwise_lambda_fn.value()
            epilogue[Self.c_type, 1](
                IndexList[2](idx // n, idx % n), SIMD[Self.c_type, 1](y)
            )
        else:
            c_ptr[idx] = y


# === Host-side launchers (standalone for testing) ========================== #


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
    force_split_k: Optional[Bool] = None,
) raises:
    """Enqueue `AppleM5MatMul.run` on the given device context.

    Accepts row-major TileTensor operands. For `transpose_b=True`, B is expected
    with shape `(N, K)`.

    `force_split_k` picks the K-reduction strategy: `None` (default) auto-routes
    under-occupied shapes (few output tiles, deep K) to split-K; `True` always
    uses split-K; `False` always uses the single-pass kernel.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
        M1-M4 lack GPU `neural accelerator`; future generations require
        re-validation.
    """
    comptime MM = AppleM5MatMul[
        in_type, c_type, transpose_b, elementwise_lambda_fn
    ]

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
            Int(b.dim[0]()) == n, "transpose_b=True expects B shape (N, K)"
        )
        debug_assert(
            Int(b.dim[1]()) == k, "transpose_b=True expects B shape (N, K)"
        )
    else:
        debug_assert(
            Int(b.dim[0]()) == k, "transpose_b=False expects B shape (K, N)"
        )
        debug_assert(
            Int(b.dim[1]()) == n, "transpose_b=False expects B shape (K, N)"
        )

    # MmaOpApple narrows the row stride to UInt16 (see `_load_fragment` /
    # `_store_fragment` in linalg/arch/apple/mma.mojo). Catch the wrap here:
    # NN A-slab stride = K, NN B-slab stride = N; NT B-slab stride = K (covered).
    debug_assert(k <= 65535, "Apple matmul: K must fit in UInt16; got K=", k)
    comptime if not transpose_b:
        debug_assert(
            n <= 65535, "Apple matmul (NN): N must fit in UInt16; got N=", n
        )

    # Per-axis next-pow2 grid for rectangular Z-order. e.g. 32x224 (Llama-3
    # MLP up-proj) -> 32x256 = 8192 launches vs the prior square 256x256 = 65536.
    var grid_m = (m + MM.BM - 1) // MM.BM
    var grid_n = (n + MM.BN - 1) // MM.BN

    # Split-K routing. `force_split_k` overrides the heuristic; when unset, the
    # heuristic routes under-occupied shapes -- few 64x64 output tiles but deep
    # K -- to split-K. The single-pass kernel launches one threadgroup per tile,
    # so a tiny M*N with large K leaves most of the GPU idle; splitting K
    # recovers 1.4-2.9x there (measured, M5 Max). The threshold is conservative;
    # normal shapes (many tiles) take the single-pass launch below.
    var tiles = grid_m * grid_n
    var num_strips = (k + MM.BK - 1) // MM.BK
    var route_split_k = force_split_k.value() if force_split_k else (
        tiles <= 16 and num_strips >= 32 and num_strips >= 8 * tiles
    )
    if route_split_k:
        var hint = max(2, min(num_strips // 4, 64 // tiles))
        enqueue_apple_matmul_split_k[
            in_type=in_type,
            c_type=c_type,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, ctx, hint)
        return

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

    comptime kernel = MM.run[
        type_of(c).LayoutType, type_of(a).LayoutType, type_of(b).LayoutType
    ]
    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        log2_m,
        log2_n,
        grid_dim=(grid_dim),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


@always_inline
def enqueue_apple_matmul_split_k[
    in_type: DType,
    c_type: DType = DType.float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    b: TileTensor[in_type, ...],
    ctx: DeviceContext,
    num_splits_hint: Int = 4,
) raises:
    """Split-K Apple M5 matmul: partition K, accumulate partials, reduce.

    `num_splits_hint` is an upper bound; the actual split count is capped so no
    split is empty (`actual_splits = ceil(num_strips / strips_per_split)` where
    `strips_per_split = ceil(num_strips / num_splits_hint)`). Best for large-K,
    small-M*N shapes where the single-pass kernel under-occupies the GPU.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
    """
    comptime MM = AppleM5MatMul[
        in_type, c_type, transpose_b, elementwise_lambda_fn
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_apple_matmul_split_k requires Apple M5"
                " (compute_capability == 5); got "
            ),
            cc,
        )

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(k <= 65535, "Apple matmul: K must fit in UInt16; got K=", k)
    comptime if not transpose_b:
        debug_assert(
            n <= 65535, "Apple matmul (NN): N must fit in UInt16; got N=", n
        )

    var grid_m = (m + MM.BM - 1) // MM.BM
    var grid_n = (n + MM.BN - 1) // MM.BN
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

    # Cap splits so none is empty: distribute BK-strips evenly (round up), then
    # recompute how many splits that actually fills.
    var num_strips = (k + MM.BK - 1) // MM.BK
    var hint = max(1, num_splits_hint)
    var strips_per_split = (num_strips + hint - 1) // hint
    var actual_splits = (num_strips + strips_per_split - 1) // strips_per_split
    var k_per_split = strips_per_split * MM.BK

    var partials = ctx.enqueue_create_buffer[DType.float32](
        actual_splits * m * n
    )

    comptime partial_kernel = MM.run_split_k_partial[
        type_of(a).LayoutType, type_of(b).LayoutType
    ]
    ctx.enqueue_function[partial_kernel](
        partials.unsafe_ptr(),
        a,
        b,
        log2_m,
        log2_n,
        k_per_split,
        grid_dim=(side_m * side_n * actual_splits),
        block_dim=(MM.THREADS_PER_BLOCK),
    )

    comptime reduce_kernel = MM.run_split_k_reduce[type_of(c).LayoutType]
    var n_elems = m * n
    ctx.enqueue_function[reduce_kernel](
        c,
        partials.unsafe_ptr(),
        actual_splits,
        grid_dim=((n_elems + MM.REDUCE_BLOCK - 1) // MM.REDUCE_BLOCK),
        block_dim=(MM.REDUCE_BLOCK),
    )
    # Keep the workspace alive until both launches are enqueued.
    _ = partials^
