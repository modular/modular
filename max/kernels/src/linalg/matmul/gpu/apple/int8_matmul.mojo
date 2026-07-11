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
"""Apple M5 int8 W8A8 matmul: `out = dequant(A_int8 @ B_int8^T)`.

Apple silicon GPU (Metal 4, `compute_capability == 5`). The int32-accumulate
sibling of the W4A16 kernel (`matmul2d_fp4.mojo` / `fp4_matmul.mojo`): where the
FP4 path decodes a packed 4-bit weight to bf16 and feeds the *float* simdgroup
MMA, this path feeds int8 A and int8 B straight to the M5 **integer widening**
simdgroup MMA (`_mma_apple` with int8 inputs -> int32 accumulator, emitting
`air.simdgroup_matrix_16x16x16_widening_multiply_accumulate`). That widening op
is a genuinely faster M5 datapath than the float MMA (~1.7-1.9x at the FLUX.2
projection/MLP shapes; the dequant epilogue nets ~1.5-1.7x). The dense
`AppleM5MatMul` cannot be reused: it hardcodes a *float* accumulator
(`MmaOpApple[float32, in_type]`) and asserts a float `c_type`.

Quantization (W8A8): both operands are symmetric-absmax int8 (`scale =
absmax / 127`), A per-row (per-token), B per-column (per-output-channel). The
kernel accumulates the raw int8xint8 products in int32, then the epilogue
DEQUANTIZES each output element by `a_scale[row] * b_scale[col]` and casts to
`c_type` (bf16), with an optional per-column bias added after dequant. A is
quantized online by `enqueue_apple_int8_quantize_activation` (a separate one-pass
kernel, reusing the `_quantize_a_block`-style absmax); B is pre-quantized at
weight load.

Structure mirrors `AppleM5MatMul._run_gemm_body`: 64x64 threadgroup block, 4
simdgroups (128 threads), each a 32x32 (2x2 `MmaOpApple`) subtile;
rectangular-Morton tile schedule (wide-N shapes are Morton-critical -- a flat
grid measured 50-70 Tops/s vs 96-99 with Morton); DRAM->register loads (no SMEM
staging -- the M5 rule). The K-loop feeds `MmaOpApple.mma()` per BK strip; edge
tiles (ragged M/N, K tail) take the bounded path.
"""

from std.gpu import WARP_SIZE, barrier, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import round
from std.memory import stack_allocation
from std.utils import IndexList

from layout import TileTensor
from layout.coord import Coord
from layout.tile_layout import TensorLayout, row_major

from linalg.arch.apple.mma import MmaOpApple


def _require_apple_m5(ctx: DeviceContext) raises:
    """Runtime guard: reject a non-Apple-M5 GPU at the host enqueue entry.

    The kernel body uses the Apple simdgroup MMA (`_mma_apple`), which compiles
    for any Apple GPU target, so the Metal-4 / M5 requirement is enforced at
    launch (matching `enqueue_apple_matmul` / `enqueue_apple_fp4_matmul`), not
    with a comptime assert that would break a non-M5 Apple build.
    """
    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "Apple int8 W8A8 matmul requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
        )


@fieldwise_init
struct Int8DequantWriter[
    c_origin: MutOrigin,
    s_origin: ImmutOrigin,
    //,
    c_type: DType,
    c_layout: TensorLayout,
    as_layout: TensorLayout,
    bs_layout: TensorLayout,
    bias_layout: TensorLayout,
    *,
    has_bias: Bool,
](ImplicitlyCopyable, Movable):
    """Owns the C-output write of the W8A8 epilogue (the write-side counterpart
    to the FP4 `Fp4WeightLoader`).

    Encapsulates the int32-accumulator -> dequant -> `c_type` store: given a
    16x16 accumulator fragment and its tile origin, it applies the per-row
    activation scale x per-column weight scale (+ optional per-column bias) and
    stores each in-bounds element through the C `TileTensor`. All addressing is
    TileTensor indexing / `store` -- no raw pointer arithmetic and no origin
    rebase. The struct is generic over the view origins (`c_origin` mutable for
    the C write, `s_origin` immutable and shared by the scale/bias reads),
    inferred by `@fieldwise_init` from the constructor args, so the fields hold
    the caller's real origins directly (a plain fieldwise constructor, no
    origin-erasing helper needed). Per the KB TileIO pattern
    (`new-primitives/amd-tile-io-expert-objects`), every register->DRAM
    transition has an owner; this is that owner for the int8 GEMM.

    The `_apple_frag_layout` maps a lane's 8-element fragment to rows
    `{rb, rb+8}` x cols `{cb, cb+1, cb+2, cb+3}`: `frag[0:4]` = row `rb`,
    `frag[4:8]` = row `rb+8` (matching `MmaOpApple._store_fragment`).
    """

    var c: TileTensor[Self.c_type, Self.c_layout, Self.c_origin]
    var a_scale: TileTensor[DType.float32, Self.as_layout, Self.s_origin]
    var b_scale: TileTensor[DType.float32, Self.bs_layout, Self.s_origin]
    var bias: TileTensor[Self.c_type, Self.bias_layout, Self.s_origin]
    var M: Int
    var N: Int

    @always_inline
    def write_frag(
        self,
        frag: SIMD[DType.int32, 8],
        tile_r0: Int,
        tile_c0: Int,
        rb: Int,
        cb: Int,
    ):
        """Dequant + store one 16x16 accumulator fragment (this lane's 8 elems).

        Bounds-guards rows `< M` and cols `< N` for the partial last tile.
        """
        comptime for half in range(2):
            var r = tile_r0 + rb + half * 8
            if r < self.M:
                var asc = self.a_scale[r]
                comptime for cc in range(4):
                    var col = tile_c0 + cb + cc
                    if col < self.N:
                        var v = (
                            Float32(frag[half * 4 + cc])
                            * asc
                            * self.b_scale[col]
                        )
                        var y = v.cast[Self.c_type]()
                        comptime if Self.has_bias:
                            y = y + self.bias[col]
                        self.c.store[width=1](
                            Coord(r, col), SIMD[Self.c_type, 1](y)
                        )


struct AppleM5Int8MatMul[
    c_type: DType = DType.bfloat16,
    *,
    has_bias: Bool = False,
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 32,
]:
    """W8A8 GEMM: int8 A x int8 B^T -> int32 accum -> per-row/col dequant.

    Parameters:
        c_type: Output element type (bf16 / fp16 / fp32). Accumulation is int32;
            the dequant multiply is done in fp32 then cast to `c_type`.
        has_bias: If True, add a per-output-column bias (in `c_type`) after
            dequant.
        BM: Threadgroup M-tile height (multiple of `SG_M`).
        BN: Threadgroup N-tile width (multiple of `SG_N`).
        BK: K-strip depth per accumulate step (multiple of `MMA_K` = 16). One
            `mma()` call iterates `BK // 16` MMA K-steps.

    `run` is the GPU kernel entry (TileTensor operands + `M`/`N`/`K`). Launch via
    `enqueue_apple_int8_matmul`.
    """

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 16
    comptime SG_M = 32  # simdgroup subtile rows (2 * MMA_M)
    comptime SG_N = 32  # simdgroup subtile cols (2 * MMA_N)
    comptime NUM_MMA_M = Self.SG_M // Self.MMA_M  # 2
    comptime NUM_MMA_N = Self.SG_N // Self.MMA_N  # 2
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE

    # int8 -> int32 widening MMA (2x2 accumulators, the M5 register optimum).
    comptime Mma = MmaOpApple[
        DType.int32,
        DType.int8,
        num_m_mmas=Self.NUM_MMA_M,
        num_n_mmas=Self.NUM_MMA_N,
        transpose_b=True,
    ]

    # === Morton (Z-order) tile scheduling (copied from `AppleM5MatMul`; wide-N
    # locality lever) ====================================================== #

    @staticmethod
    def morton_decode_2d(flat_idx: UInt32) -> Tuple[UInt32, UInt32]:
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
        flat_idx: UInt32, log2_m: UInt32, log2_n: UInt32
    ) -> Tuple[UInt32, UInt32]:
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

    @__name(t"apple_int8_matmul_run_{Self.c_type}_{Self.has_bias}")
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        as_layout: TensorLayout,
        bs_layout: TensorLayout,
        bias_layout: TensorLayout,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
        a: TileTensor[DType.int8, a_layout, ImmutAnyOrigin],
        b: TileTensor[DType.int8, b_layout, ImmutAnyOrigin],
        a_scale: TileTensor[DType.float32, as_layout, ImmutAnyOrigin],
        b_scale: TileTensor[DType.float32, bs_layout, ImmutAnyOrigin],
        bias: TileTensor[Self.c_type, bias_layout, ImmutAnyOrigin],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """W8A8 kernel entry. C `(M, N)`, A `(M, K)` int8, B `(N, K)` int8
        (`transpose_b`), `a_scale` `(M,)`, `b_scale` `(N,)`, `bias` `(N,)` (used
        iff `has_bias`). Grid is `(1<<log2_grid_m) * (1<<log2_grid_n)`
        threadgroups; OOB threadgroups early-return after Morton decode.
        """
        var m = Int32(c.dim[0]())
        var n = Int32(c.dim[1]())
        var k = Int(a.dim[1]())

        comptime BM_i = Int32(Self.BM)
        comptime BN_i = Int32(Self.BN)
        comptime SG_M_i = Int32(Self.SG_M)
        comptime SG_N_i = Int32(Self.SG_N)

        var grid_m = (m + BM_i - 1) // BM_i
        var grid_n = (n + BN_i - 1) // BN_i

        var tile_mn = Self.morton_decode_2d_rect(
            UInt32(block_idx.x), log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // Int32(Self.NUM_SG_N)
        var sg_n_idx = sg_id % Int32(Self.NUM_SG_N)

        var row_base = tile_m * BM_i + sg_m_idx * SG_M_i
        var col_base = tile_n * BN_i + sg_n_idx * SG_N_i
        if row_base >= m or col_base >= n:
            return

        # Absolute simdgroup-subtile indices (in SG_M / SG_N units).
        var sg_row = Int(row_base // SG_M_i)
        var sg_col = Int(col_base // SG_N_i)

        var mma_op = Self.Mma()
        var accum = Self.Mma.zero_accum()

        var is_edge = (row_base + SG_M_i > m) or (col_base + SG_N_i > n)
        var n_full_strips = k // Self.BK
        var has_k_tail = (k % Self.BK) != 0

        # K-loop: feed int8 A/B strips to the widening MMA. Interior tiles use
        # the fast unbounded path; edge tiles (ragged M/N or K tail) zero-fill
        # OOB via the bounded path (`0 * int == 0`, no NaN concern for ints).
        @parameter
        def _accumulate[bounded: Bool]():
            var valid_rows = Int(min(SG_M_i, m - row_base))
            var valid_cols = Int(min(SG_N_i, n - col_base))
            for ks in range(n_full_strips):
                var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                comptime if bounded:
                    mma_op.mma[bounded=True](
                        accum,
                        a_strip,
                        b_strip,
                        a_valid_rows=valid_rows,
                        b_valid_cols=valid_cols,
                        k_valid=Self.BK,
                    )
                else:
                    mma_op.mma(accum, a_strip, b_strip)
            comptime if bounded:
                if has_k_tail:
                    var ks = n_full_strips
                    var k_rem = k - ks * Self.BK
                    var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                    var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                    mma_op.mma[bounded=True](
                        accum,
                        a_strip,
                        b_strip,
                        a_valid_rows=valid_rows,
                        b_valid_cols=valid_cols,
                        k_valid=k_rem,
                    )

        if is_edge or has_k_tail:
            _accumulate[True]()
        else:
            _accumulate[False]()

        # === Dequant epilogue ============================================== #
        # The C-output write (int32 accum -> dequant -> `c_type` store) is owned
        # by `Int8DequantWriter`; `run` just hands it each 16x16 fragment + its
        # tile origin. Dequant elem (r, c) = int32 * a_scale[r] * b_scale[c]
        # (+ bias[c]) -- see the writer for the fragment/lane layout.
        var rb = Int(mma_op.rb)
        var cb = Int(mma_op.cb)
        var writer = Int8DequantWriter[
            Self.c_type,
            c_layout,
            as_layout,
            bs_layout,
            bias_layout,
            has_bias=Self.has_bias,
        ](c, a_scale, b_scale, bias, Int(m), Int(n))

        comptime for mi in range(Self.NUM_MMA_M):
            comptime for ni in range(Self.NUM_MMA_N):
                writer.write_frag(
                    accum[mi * Self.NUM_MMA_N + ni],
                    Int(row_base) + mi * Self.MMA_M,
                    Int(col_base) + ni * Self.MMA_N,
                    rb,
                    cb,
                )


@always_inline
def enqueue_apple_int8_matmul[
    c_type: DType = DType.bfloat16,
    *,
    has_bias: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[DType.int8, ...],
    b: TileTensor[DType.int8, ...],
    a_scale: TileTensor[DType.float32, ...],
    b_scale: TileTensor[DType.float32, ...],
    bias: TileTensor[c_type, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the W8A8 matmul: `out = dequant(a @ b^T)`.

    `a` is int8 `(M, K)`, `b` int8 `(N, K)` (`transpose_b`), `a_scale` `(M,)`
    per-row, `b_scale` `(N,)` per-column, `bias` `(N,)` in `c_type` (used iff
    `has_bias`; pass a length-1 dummy otherwise). C is `(M, N)`.

    Parameters:
        c_type: Output element type (bf16 / fp16 / fp32).
        has_bias: If True, add `bias[col]` after dequant.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime MM = AppleM5Int8MatMul[c_type, has_bias=has_bias]

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(b.dim[0]()) == n, "B must be (N, K)")
    debug_assert(Int(b.dim[1]()) == k, "B K must match A K")
    # MmaOpApple narrows row strides to UInt16.
    debug_assert(
        k <= 65535 and n <= 65535,
        "Apple int8 matmul: K and N must fit in UInt16",
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

    comptime kernel = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scale).LayoutType,
        type_of(b_scale).LayoutType,
        type_of(bias).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        b.as_immut(),
        a_scale.as_immut(),
        b_scale.as_immut(),
        bias.as_immut(),
        log2_m,
        log2_n,
        grid_dim=(side_m * side_n),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


# ===----------------------------------------------------------------------=== #
# Online per-row activation quantization (bf16 -> int8 + per-row fp32 scale)
# ===----------------------------------------------------------------------=== #


struct AppleInt8ActQuant[in_type: DType = DType.bfloat16, *, THREADS: Int = 64]:
    """Per-row symmetric-absmax int8 quantization of the activation.

    One threadgroup per row: the `THREADS` threads cooperatively reduce
    `absmax` over the row's K columns (strided), then quantize
    `q = roundeven(x * 127 / absmax)` and write the row's fp32 `scale =
    absmax / 127`. Mirrors `_quantize_a_block`'s scheme (symmetric absmax/127,
    no zero-point) but as a GPU row kernel. No SMEM reduction primitive is
    assumed; the absmax reduction is a two-pass strided scan (the row fits in
    L1/L2 and is re-read cheaply -- K <= 12288 for FLUX).
    """

    @__name("apple_int8_act_quant")
    @staticmethod
    def run[
        q_layout: TensorLayout, a_layout: TensorLayout, s_layout: TensorLayout
    ](
        q: TileTensor[DType.int8, q_layout, MutAnyOrigin],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin],
        a_scale: TileTensor[DType.float32, s_layout, MutAnyOrigin],
        K: Int,
    ):
        var row = Int(block_idx.x)
        var tid = Int(thread_idx.x)

        # Pass 1: per-thread partial absmax over strided columns of this row,
        # then a threadgroup all-reduce via a tiny shared-scratch max.
        var local_max = Float32(0)
        var j = tid
        while j < K:
            var v = abs(Float32(a[row, j]))
            if v > local_max:
                local_max = v
            j += Self.THREADS

        var row_max = _threadgroup_max[Self.THREADS](local_max)

        var scale = row_max / 127.0 if row_max != 0.0 else Float32(0)
        var mult = 127.0 / row_max if row_max != 0.0 else Float32(0)
        if tid == 0:
            a_scale.store[width=1](Coord(row), SIMD[DType.float32, 1](scale))

        # Pass 2: quantize.
        j = tid
        while j < K:
            var qi = Int(round(Float32(a[row, j]) * mult))
            q.store[width=1](Coord(row, j), SIMD[DType.int8, 1](qi))
            j += Self.THREADS


@always_inline
def _threadgroup_max[nthreads: Int](val: Float32) -> Float32:
    """Threadgroup all-reduce max over `nthreads` (one warp-multiple block).

    Writes each thread's value to SMEM, barriers, then thread 0's linear scan
    broadcasts the max back through SMEM. `nthreads` is small (64) so the linear
    reduction is cheap and needs no tree.
    """
    var s = stack_allocation[
        nthreads, Float32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    s[tid] = val
    barrier()
    if tid == 0:
        var m = s[0]
        for i in range(1, nthreads):
            if s[i] > m:
                m = s[i]
        s[0] = m
    barrier()
    return s[0]


@always_inline
def enqueue_apple_int8_quantize_activation[
    in_type: DType = DType.bfloat16,
](
    q: TileTensor[mut=True, DType.int8, ...],
    a: TileTensor[in_type, ...],
    a_scale: TileTensor[mut=True, DType.float32, ...],
    ctx: DeviceContext,
) raises:
    """Quantize `a` `(M, K)` to int8 `q` + per-row fp32 `a_scale` `(M,)`.

    One threadgroup per row (M threadgroups, 64 threads each). Symmetric
    absmax/127, no zero-point. Raises if the GPU is not Apple M5.
    """
    _require_apple_m5(ctx)
    var m = Int(a.dim[0]())
    var k = Int(a.dim[1]())
    comptime QK = AppleInt8ActQuant[in_type]
    comptime kernel = QK.run[
        type_of(q).LayoutType,
        type_of(a).LayoutType,
        type_of(a_scale).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        q,
        a.as_immut(),
        a_scale,
        k,
        grid_dim=(m),
        block_dim=(QK.THREADS),
    )
