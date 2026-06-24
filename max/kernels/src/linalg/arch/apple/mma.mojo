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
"""Apple Silicon MMA operation struct for TileTensor.

Simdgroup-level, register-owning MMA abstraction following the AMD MmaOp
pattern. Each simdgroup (32 threads) instantiates its own MmaOpApple.

Use mma() for interior tiles (caller guarantees in-bounds). Use
mma[bounded=True]() for edge tiles (zero-fills OOB elements). The
kernel should check once per simdgroup, not per load.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.gpu import lane_id
from std.gpu.compute.arch.mma_apple import _mma_apple_transposable
from std.sys.info import align_of

from layout import TileTensor


@fieldwise_init
struct ConvIm2colParams(
    Copyable, DevicePassable, ImplicitlyCopyable, ImplicitlyDeletable, Movable
):
    """Runtime conv geometry for the online im2col A-operand loader.

    SM100/Apple M5 (`compute_capability == 5`). The fused conv path
    (`AppleM5MatMul.run_conv`) never materialises the `[M, K]` im2col matrix;
    instead each A MMA-fragment is gathered directly from the NHWC input by the
    address math below. This struct carries the per-launch conv parameters the
    gather needs, and is `DevicePassable` (all-`Int32` POD, `device_type = Self`)
    so it can cross `enqueue_function` directly.

    The (M, K) -> NHWC map mirrors `nn/conv/gpu/im2col_matmul_2d.mojo` bit-for-bit
    so the fused result matches the materialised path:

      m -> (batch, h_out, w_out):  batch = m // HW_out;  spatial = m % HW_out;
                                    h_out = spatial // W_out; w_out = spatial % W_out
      k -> (r, s, c):              r = k // (S*C);  sc = k % (S*C);
                                    s = sc // C;  c = sc % C
      h_in = h_out*stride_h - pad_h + r;  w_in = w_out*stride_w - pad_w + s
      OOB (h_in/w_in outside [0,H)/[0,W)) -> 0
      in_idx = batch*(H*W*C) + h_in*(W*C) + w_in*C + c

    Dilation is assumed 1 (enforced at the conv dispatch gate); the map matches
    the materialiser, which also hardcodes dilation 1.
    """

    var H: Int32
    var W: Int32
    var C: Int32
    var R: Int32
    var S: Int32
    var H_out: Int32
    var W_out: Int32
    var pad_h: Int32
    var pad_w: Int32
    var stride_h: Int32
    var stride_w: Int32

    def __init__(out self):
        """Zero-init all fields, for the dense matmul path that ignores conv."""
        self.H = 0
        self.W = 0
        self.C = 0
        self.R = 0
        self.S = 0
        self.H_out = 0
        self.W_out = 0
        self.pad_h = 0
        self.pad_w = 0
        self.stride_h = 0
        self.stride_w = 0

    # `DevicePassable`: identity device form (all-Int32 POD, bit-copied).
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ConvIm2colParams"


struct MmaOpApple[
    out_type: DType,
    in_type: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    *,
    b_type: DType = in_type,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
]:
    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 16
    comptime FRAG_SIZE = 8
    comptime num_accum = Self.num_m_mmas * Self.num_n_mmas
    comptime AccumType = InlineArray[
        SIMD[Self.out_type, Self.FRAG_SIZE], Self.num_accum
    ]

    var rb: Int
    var cb: Int

    @always_inline
    def __init__(out self):
        var lid = Int(lane_id())
        self.rb = ((lid & 7) >> 1) + ((lid & 16) >> 2)
        self.cb = ((lid & 1) << 2) + (lid & 8)

    @staticmethod
    @always_inline
    def zero_accum() -> Self.AccumType:
        return Self.AccumType(fill=SIMD[Self.out_type, Self.FRAG_SIZE](0))

    @staticmethod
    def _row_stride(tile: TileTensor) -> Int:
        """Return the non-unit stride (row stride for fragment layout).

        For row-major (stride[1]=1): returns stride[0].
        For col-major (stride[0]=1): returns stride[1].
        Asserts at comptime that exactly one stride is 1.
        """
        comptime if type_of(tile).static_stride[1] == 1:
            return Int(tile.layout.stride[0]().value())
        elif type_of(tile).static_stride[0] == 1:
            return Int(tile.layout.stride[1]().value())
        else:
            comptime assert (
                False
            ), "Tile must have a contiguous dimension (static_stride == 1)"

    @always_inline
    def _load_fragment[
        dtype: DType
    ](
        self,
        tile: TileTensor[dtype, ...],
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[
        dtype, 8
    ]:
        # Offsets are precomputed + hoisted once per slab in `mma()`.
        # Element alignment only: `rb * row_stride` is unaligned for odd K/N.
        # `width` drives vectorization, not the alignment hint.
        comptime alignment = align_of[Scalar[dtype]]()
        var lo = (tile.ptr + lo_off).load[width=4, alignment=alignment]()
        var hi = (tile.ptr + hi_off).load[width=4, alignment=alignment]()
        return lo.join(hi)

    @always_inline
    def _store_fragment[
        dtype: DType
    ](self, tile: TileTensor[mut=True, dtype, ...], frag: SIMD[dtype, 8],):
        var row_stride = Self._row_stride(tile)
        var off_lo = self.rb * row_stride + self.cb
        var off_hi = (self.rb + 8) * row_stride + self.cb
        (tile.ptr + off_lo).store(frag.slice[4, offset=0]())
        (tile.ptr + off_hi).store(frag.slice[4, offset=4]())

    @always_inline
    def load_fragment[
        dtype: DType, bounded: Bool = False
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int = 16,
    ) -> SIMD[
        dtype, 8
    ]:
        """Loads one 16x16 simdgroup fragment from a TileTensor sub-tile.

        The `_apple_frag_layout` bit-scatter is owned here at the `MmaOpApple`
        layer, not via a TileTensor `distribute` (KB
        `exceptions/apple-mma-fragment-is-not-distribute-expressible`). Used for
        the V operand of a P.V MMA, where A (P) is a register-resident score
        fragment and so cannot go through `mma()`.

        `bounded=True` zero-fills rows `>= valid_rows` instead of reading them
        -- needed for the V load on the last KV sub-tile when `num_keys % 16 !=
        0`, where reading an OOB row makes `0 * V_oob == NaN` and poisons the
        fp32 PV accumulator (KB `kernels/apple-m5-fa-prefill`). Depth columns are
        always in-bounds, so only rows are predicated.

        Parameters:
            dtype: The element dtype of the source tile.
            bounded: When True, zero-fill rows `>= valid_rows`.

        Args:
            tile: A 16x16 source view (row- or col-major).
            valid_rows: Valid rows from the sub-tile origin (only when bounded).

        Returns:
            This lane's 8-element fragment.
        """
        var row_stride = Self._row_stride(tile)
        var lo_off = self.rb * row_stride + self.cb
        var hi_off = (self.rb + 8) * row_stride + self.cb
        comptime if bounded:
            # All 16 cols valid (depth is contiguous within a token); only the
            # key (row) axis can run past the valid KV length.
            return self._bounded_load[dtype](
                tile, valid_rows, 16, lo_off, hi_off
            )
        else:
            return self._load_fragment[dtype](tile, lo_off, hi_off)

    @always_inline
    def _do_load[
        dtype: DType, bounded: Bool
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[dtype, 8]:
        """Dispatch fast or bounded load; swap bounds for col-major tiles."""
        comptime if bounded:
            comptime col_major = type_of(tile).static_stride[0] == 1
            comptime if col_major:
                return self._bounded_load[dtype](
                    tile, valid_cols, valid_rows, lo_off, hi_off
                )
            else:
                return self._bounded_load[dtype](
                    tile, valid_rows, valid_cols, lo_off, hi_off
                )
        else:
            return self._load_fragment[dtype](tile, lo_off, hi_off)

    @always_inline
    def _bounded_load[
        dtype: DType
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[dtype, 8]:
        """Bounded path: predicated loads, zero-fill for OOB.

        Vectorized 4-wide when all 4 cols in bounds; scalar per-element
        when straddling the column boundary. Zero-filled OOB elements
        contribute nothing to the dot product.
        """
        var col = self.cb
        var lo = SIMD[dtype, 4](0)
        var hi = SIMD[dtype, 4](0)

        if self.rb < valid_rows:
            if col + 3 < valid_cols:
                lo = (tile.ptr + lo_off).load[width=4]()
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        lo[i] = tile.ptr[lo_off + i]

        if self.rb + 8 < valid_rows:
            if col + 3 < valid_cols:
                hi = (tile.ptr + hi_off).load[width=4]()
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        hi[i] = tile.ptr[hi_off + i]

        return lo.join(hi)

    @always_inline
    def _bounded_store[
        dtype: DType
    ](
        self,
        tile: TileTensor[mut=True, dtype, ...],
        frag: SIMD[dtype, 8],
        valid_rows: Int,
        valid_cols: Int,
    ):
        """Bounded path: predicated stores, skip OOB elements.

        Vectorized 4-wide when all 4 cols in bounds; scalar per-element
        when straddling the column boundary.
        """
        var row_stride = Self._row_stride(tile)
        var col = self.cb

        if self.rb < valid_rows:
            var off = self.rb * row_stride + col
            if col + 3 < valid_cols:
                (tile.ptr + off).store(frag.slice[4, offset=0]())
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        tile.ptr[off + i] = frag[i]

        if self.rb + 8 < valid_rows:
            var off = (self.rb + 8) * row_stride + col
            if col + 3 < valid_cols:
                (tile.ptr + off).store(frag.slice[4, offset=4]())
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        tile.ptr[off + i] = frag[4 + i]

    @always_inline
    def mma[
        bounded: Bool = False
    ](
        self,
        mut accum: Self.AccumType,
        a_tile: TileTensor[Self.in_type, ...],
        b_tile: TileTensor[Self.b_type, ...],
        a_valid_rows: Int = Self.num_m_mmas * 16,
        b_valid_cols: Int = Self.num_n_mmas * 16,
        k_valid: Int = type_of(a_tile).static_shape[1],
    ):
        """Process K elements across all M/N tile positions.

        The K depth is inferred from the A tile's column dimension and
        must be a multiple of 16. For K=16 this is one MMA step; for
        K=32 this is two steps, etc. The struct iterates K internally.

        Tiles may be row-major or col-major. The stride layout is
        detected from static_stride and the hardware transpose flag is
        derived via XOR with the transpose parameter:
        hw_flag = is_col_major XOR transpose_param.

        Use mma() (bounded=False) for interior tiles where all memory
        is in-bounds. Use mma[bounded=True]() for edge tiles --
        zero-fills OOB elements. The kernel should check once per
        simdgroup, not per load.

        Args:
            accum: Caller-owned InlineArray of SIMD[out_type, 8]
                accumulators, one per (num_m_mmas * num_n_mmas) tile.
            a_tile: A operand, shape (num_m_mmas * 16, K).
            b_tile: B operand, shape (K, num_n_mmas * 16) or
                (num_n_mmas * 16, K) if transpose_b.
            a_valid_rows: Valid rows from tile origin (bounded path only).
            b_valid_cols: Valid cols from tile origin (bounded path only).
            k_valid: Valid K elements across all steps (bounded path
                only). Defaults to the tile's full K dimension.
        """
        # Extract logical dimensions accounting for transpose.
        # A stored as (M, K) normally, (K, M) when transpose_a.
        # B stored as (K, N) normally, (N, K) when transpose_b.
        comptime a_k = (
            type_of(a_tile)
            .static_shape[0] if Self.transpose_a else type_of(a_tile)
            .static_shape[1]
        )
        comptime a_m = (
            type_of(a_tile)
            .static_shape[1] if Self.transpose_a else type_of(a_tile)
            .static_shape[0]
        )
        comptime b_k = (
            type_of(b_tile)
            .static_shape[1] if Self.transpose_b else type_of(b_tile)
            .static_shape[0]
        )
        comptime b_n = (
            type_of(b_tile)
            .static_shape[0] if Self.transpose_b else type_of(b_tile)
            .static_shape[1]
        )
        comptime assert a_k == b_k, "A and B K dimensions must match"
        comptime assert (
            a_k % Self.MMA_K == 0
        ), "K dimension must be a multiple of 16"
        comptime assert (
            a_m % Self.MMA_M == 0
        ), "A M dimension must be a multiple of 16"
        comptime assert (
            b_n % Self.MMA_N == 0
        ), "B N dimension must be a multiple of 16"

        # Hardware transpose = is_col_major XOR transpose_param.
        # _row_stride already asserts one stride is 1 at comptime.
        comptime a_col_major = type_of(a_tile).static_stride[0] == 1
        comptime b_col_major = type_of(b_tile).static_stride[0] == 1
        comptime hw_transpose_a = a_col_major != Self.transpose_a
        comptime hw_transpose_b = b_col_major != Self.transpose_b

        comptime num_k_steps = a_k // Self.MMA_K

        # Precompute the per-lane offsets once per slab.
        var a_row_stride = Self._row_stride(a_tile)
        var b_row_stride = Self._row_stride(b_tile)
        var rb_plus_8 = self.rb + 8
        var a_lo_off = self.rb * a_row_stride + self.cb
        var a_hi_off = rb_plus_8 * a_row_stride + self.cb
        var b_lo_off = self.rb * b_row_stride + self.cb
        var b_hi_off = rb_plus_8 * b_row_stride + self.cb

        comptime for ki in range(num_k_steps):
            # Pre-load B fragments for this K-step.
            var b_frags = InlineArray[
                SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
            ](uninitialized=True)
            comptime for ni in range(Self.num_n_mmas):
                var b_sub = b_tile.tile[16, 16](
                    ni if Self.transpose_b else ki,
                    ki if Self.transpose_b else ni,
                )
                b_frags[ni] = self._do_load[Self.b_type, bounded](
                    b_sub,
                    (b_valid_cols - ni * 16) if Self.transpose_b else (
                        k_valid - ki * 16
                    ),
                    (k_valid - ki * 16) if Self.transpose_b else (
                        b_valid_cols - ni * 16
                    ),
                    b_lo_off,
                    b_hi_off,
                )

            comptime for mi in range(Self.num_m_mmas):
                var a_sub = a_tile.tile[16, 16](
                    ki if Self.transpose_a else mi,
                    mi if Self.transpose_a else ki,
                )
                var a_frag = self._do_load[Self.in_type, bounded](
                    a_sub,
                    (k_valid - ki * 16) if Self.transpose_a else (
                        a_valid_rows - mi * 16
                    ),
                    (a_valid_rows - mi * 16) if Self.transpose_a else (
                        k_valid - ki * 16
                    ),
                    a_lo_off,
                    a_hi_off,
                )

                comptime for ni in range(Self.num_n_mmas):
                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        a_frag,
                        b_frags[ni],
                        accum[mi * Self.num_n_mmas + ni],
                        hw_transpose_a,
                        hw_transpose_b,
                    )

    @always_inline
    def _load_a_im2col_fragment[
        input_origin: ImmutOrigin
    ](
        self,
        input_ptr: UnsafePointer[Scalar[Self.in_type], input_origin],
        conv: ConvIm2colParams,
        m_base: Int32,
        k_base: Int32,
        m_valid: Int32,
        k_total: Int32,
    ) -> SIMD[Self.in_type, Self.FRAG_SIZE]:
        """Gather one 16x16 A MMA-fragment directly from NHWC input (no scratch).

        Reproduces the `[M, K]` im2col element at the lane's two rows
        (`m_base + rb`, `m_base + rb + 8`) and four columns
        (`k_base + cb .. k_base + cb + 3`), gathering each from NHWC source with
        OOB zero-fill -- both the conv halo (`h_in`/`w_in` outside the image) and
        the matmul tile edge (`row >= m_valid`, or absolute `k_base + col >=
        k_total` for the partial-K tail, mirroring `_bounded_load`). The
        (M, K) -> NHWC map matches `nn/conv/gpu/im2col_matmul_2d.mojo` exactly.
        fp32 index math in Int32 (Apple scalar ALU is faster on 32-bit; see KB
        `patterns/apple-m5-gpu-performance-considerations`).

        Args:
            input_ptr: NHWC input base pointer.
            conv: Conv geometry for the gather.
            m_base: Absolute M row of this 16x16 sub-tile's origin.
            k_base: Absolute K column of this 16x16 sub-tile's origin.
            m_valid: Valid M rows from `m_base` (ragged-M edge).
            k_total: Total K extent (R*S*C); cols at or past it zero-fill.
        """
        var HW_out = conv.H_out * conv.W_out
        var SC = conv.S * conv.C
        var hwc = conv.H * conv.W * conv.C
        var wc = conv.W * conv.C

        var frag = SIMD[Self.in_type, Self.FRAG_SIZE](0)

        # `rb`/`cb` are stored as `Int`; do the gather math in Int32 (Apple
        # scalar ALU is faster on 32-bit -- KB apple-m5-gpu-performance-...).
        var rb_i32 = Int32(self.rb)
        var cb_i32 = Int32(self.cb)

        # The lane owns rows {rb, rb+8} (lo packed at frag[0:4], hi at frag[4:8])
        # x 4 contiguous cols {cb .. cb+3} of the 16x16 A sub-tile, matching the
        # `_apple_frag_layout` bit-scatter that `_load_fragment` reads for the
        # linear A operand (KB exceptions/apple-mma-fragment-is-not-distribute-
        # expressible). Two rows, unrolled (no mutable-capture closure).
        comptime for half in range(2):
            comptime out_off = half * 4
            var row = rb_i32 + Int32(half * 8)
            # Tile-edge M bound: zero-fill OOB rows (matmul ragged-M tail).
            if row >= m_valid:
                continue
            var m = m_base + row
            # m -> (batch, h_out, w_out)
            var batch = m // HW_out
            var spatial = m % HW_out
            var h_out = spatial // conv.W_out
            var w_out = spatial % conv.W_out
            var h_base = h_out * conv.stride_h - conv.pad_h
            var w_base = w_out * conv.stride_w - conv.pad_w
            var batch_base = batch * hwc

            # Coalesced fast path: the 4 columns `k_base+cb+0..3` are CONSECUTIVE
            # k, so `c = k % C` is consecutive and the 4 NHWC source addresses
            # `in_idx = batch_base + h_in*wc + w_in*C + c` are CONTIGUOUS (differ
            # only in c). `r,s` (hence `h_in,w_in` and halo validity) are
            # IDENTICAL across the 4 iff the run stays within one c-block
            # (`c0+3 < C`) and within k_total (`k0+3 < k_total`). Then compute
            # r,s,h_in,w_in and the halo check ONCE and issue a SINGLE width-4
            # load -- coalesced, no per-element div/mod (KB
            # apple-m5-gpu-performance-considerations). Element alignment only
            # (in_idx may be unaligned for C%4!=0); width drives vectorization,
            # not the alignment hint, so this never faults
            # (KB tiletensor-vectorized-store-alignment-odd-stride).
            var k0 = k_base + cb_i32
            var c0 = k0 % conv.C
            if c0 + 3 < conv.C and k0 + 3 < k_total:
                # Whole 4-run is one contiguous c-stripe in a single (r,s) tap.
                var r = k0 // SC
                var sc = k0 % SC
                var s = sc // conv.C
                var h_in = h_base + r
                var w_in = w_base + s
                # Conv halo: zero-fill (all 4) if the source pixel is OOB.
                if h_in >= 0 and h_in < conv.H and w_in >= 0 and w_in < conv.W:
                    var in_idx = batch_base + h_in * wc + w_in * conv.C + c0
                    comptime in_align = align_of[Scalar[Self.in_type]]()
                    var v = (input_ptr + Int(in_idx)).load[
                        width=4, alignment=in_align
                    ]()
                    frag[out_off + 0] = v[0]
                    frag[out_off + 1] = v[1]
                    frag[out_off + 2] = v[2]
                    frag[out_off + 3] = v[3]
                continue

            # Slow path: the 4-run straddles a c-block boundary (`c0+3 >= C`) or
            # the partial-K tail (`k0+3 >= k_total`). Fall back to per-element
            # scalar gather (each col may map to a different r,s,c / validity).
            for i in range(4):
                var k = k0 + Int32(i)
                # Tile-edge K bound: zero-fill OOB cols (partial-K tail).
                if k >= k_total:
                    continue
                # k -> (r, s, c)
                var r = k // SC
                var sc = k % SC
                var s = sc // conv.C
                var c = sc % conv.C
                var h_in = h_base + r
                var w_in = w_base + s
                # Conv halo: zero-fill source pixels outside the image.
                if h_in >= 0 and h_in < conv.H and w_in >= 0 and w_in < conv.W:
                    var in_idx = batch_base + h_in * wc + w_in * conv.C + c
                    frag[out_off + i] = input_ptr[Int(in_idx)]
        return frag

    @always_inline
    def mma_im2col[
        input_origin: ImmutOrigin
    ](
        self,
        mut accum: Self.AccumType,
        input_ptr: UnsafePointer[Scalar[Self.in_type], input_origin],
        conv: ConvIm2colParams,
        b_tile: TileTensor[Self.b_type, ...],
        m_base: Int32,
        k_base: Int32,
        m_valid: Int32,
        k_total: Int32,
        b_valid_cols: Int,
        k_valid: Int,
    ):
        """Fused-conv MMA step: A from online im2col gather, B as in `mma`.

        SM100/Apple M5. Mirrors `mma[bounded=True]` for the B-operand load and
        the accumulate, but produces the A MMA-fragment by gathering NHWC input
        in-line (`_load_a_im2col_fragment`) instead of reading a contiguous
        `[M, K]` tile. The conv A operand is logically `(M, K)` row-major and
        never transposed, so `hw_transpose_a` is False (matches `mma`'s
        derivation for a row-major, `transpose_a=False` A).

        Args:
            accum: Caller-owned accumulators (one per num_m_mmas * num_n_mmas).
            input_ptr: NHWC input base pointer.
            conv: Conv geometry for the im2col gather.
            b_tile: B operand, `(K, num_n_mmas*16)` or `(num_n_mmas*16, K)` if
                transpose_b -- same contract as `mma`.
            m_base: Absolute M (output-pixel) row of this simdgroup's A tile.
            k_base: Absolute K (R*S*C) column where this BK strip starts.
            m_valid: Valid M rows from `m_base` (ragged-M edge zero-fill).
            k_total: Total K extent (R*S*C); A cols at or past it zero-fill.
            b_valid_cols: Valid N cols from the B tile origin (edge zero-fill).
            k_valid: Valid K elements in this strip (partial-K tail zero-fill).
        """
        comptime b_k = (
            type_of(b_tile)
            .static_shape[1] if Self.transpose_b else type_of(b_tile)
            .static_shape[0]
        )
        comptime b_n = (
            type_of(b_tile)
            .static_shape[0] if Self.transpose_b else type_of(b_tile)
            .static_shape[1]
        )
        comptime assert (
            b_k % Self.MMA_K == 0
        ), "K dimension must be a multiple of 16"
        comptime assert (
            b_n % Self.MMA_N == 0
        ), "B N dimension must be a multiple of 16"

        comptime b_col_major = type_of(b_tile).static_stride[0] == 1
        comptime hw_transpose_b = b_col_major != Self.transpose_b
        # A is im2col `(M, K)` row-major, never transposed.
        comptime hw_transpose_a = False

        comptime num_k_steps = b_k // Self.MMA_K

        var b_row_stride = Self._row_stride(b_tile)
        var rb_plus_8 = self.rb + 8
        var b_lo_off = self.rb * b_row_stride + self.cb
        var b_hi_off = rb_plus_8 * b_row_stride + self.cb

        comptime for ki in range(num_k_steps):
            # Pre-load B fragments for this K-step (identical to `mma`).
            var b_frags = InlineArray[
                SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
            ](uninitialized=True)
            comptime for ni in range(Self.num_n_mmas):
                var b_sub = b_tile.tile[16, 16](
                    ni if Self.transpose_b else ki,
                    ki if Self.transpose_b else ni,
                )
                b_frags[ni] = self._do_load[Self.b_type, bounded=True](
                    b_sub,
                    (b_valid_cols - ni * 16) if Self.transpose_b else (
                        k_valid - ki * 16
                    ),
                    (k_valid - ki * 16) if Self.transpose_b else (
                        b_valid_cols - ni * 16
                    ),
                    b_lo_off,
                    b_hi_off,
                )

            comptime for mi in range(Self.num_m_mmas):
                # A fragment gathered from NHWC for this (mi, ki) 16x16 sub-tile.
                var a_frag = self._load_a_im2col_fragment(
                    input_ptr,
                    conv,
                    m_base + Int32(mi * 16),
                    k_base + Int32(ki * 16),
                    m_valid - Int32(mi * 16),
                    k_total,
                )

                comptime for ni in range(Self.num_n_mmas):
                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        a_frag,
                        b_frags[ni],
                        accum[mi * Self.num_n_mmas + ni],
                        hw_transpose_a,
                        hw_transpose_b,
                    )

    @always_inline
    def store(
        self,
        accum: Self.AccumType,
        d_tile: TileTensor[mut=True, Self.out_type, ...],
    ):
        """Store all accumulators to output tile (unconditional).

        Caller guarantees all elements are in-bounds.
        """
        comptime for mi in range(Self.num_m_mmas):
            comptime for ni in range(Self.num_n_mmas):
                var sub = d_tile.tile[16, 16](mi, ni)
                self._store_fragment(sub, accum[mi * Self.num_n_mmas + ni])

    @always_inline
    def store_bounded(
        self,
        accum: Self.AccumType,
        d_tile: TileTensor[mut=True, Self.out_type, ...],
        valid_rows: Int,
        valid_cols: Int,
    ):
        """Stores accumulators where `(row < valid_rows) and (col < valid_cols)`.

        Assumes row-major `d_tile`; for col-major, mirror `_do_load`'s swap.
        """
        comptime for mi in range(Self.num_m_mmas):
            comptime for ni in range(Self.num_n_mmas):
                var sub = d_tile.tile[16, 16](mi, ni)
                self._bounded_store(
                    sub,
                    accum[mi * Self.num_n_mmas + ni],
                    valid_rows=valid_rows - mi * 16,
                    valid_cols=valid_cols - ni * 16,
                )
