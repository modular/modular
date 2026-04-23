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

from std.gpu import lane_id
from std.gpu.compute.arch.mma_apple import _mma_apple_transposable

from layout import TileTensor


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

    var rb: UInt16
    var cb: UInt16

    @always_inline
    def __init__(out self):
        var lid: Int = lane_id()
        self.rb = UInt16(((lid & 7) >> 1) + ((lid & 16) >> 2))
        self.cb = UInt16(((lid & 1) << 2) + (lid & 8))

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
    ](self, tile: TileTensor[dtype, ...],) -> SIMD[dtype, 8]:
        # UInt16 stride narrowing: on Apple GPU this fast 16-bit multiply
        # path closes the gap with the raw `_mma_apple` intrinsic.
        var row_stride = UInt16(Self._row_stride(tile))
        var lo = (tile.ptr + Int(self.rb * row_stride + self.cb)).load[
            width=4
        ]()
        var hi = (tile.ptr + Int((self.rb + 8) * row_stride + self.cb)).load[
            width=4
        ]()
        return lo.join(hi)

    @always_inline
    def _store_fragment[
        dtype: DType
    ](self, tile: TileTensor[mut=True, dtype, ...], frag: SIMD[dtype, 8],):
        var row_stride = UInt16(Self._row_stride(tile))
        var off_lo = self.rb * row_stride + self.cb
        var off_hi = (self.rb + 8) * row_stride + self.cb
        (tile.ptr + Int(off_lo)).store(frag.slice[4, offset=0]())
        (tile.ptr + Int(off_hi)).store(frag.slice[4, offset=4]())

    @always_inline
    def _do_load[
        dtype: DType, bounded: Bool
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
    ) -> SIMD[dtype, 8]:
        """Dispatch to fast or bounded load based on comptime flag."""
        comptime if bounded:
            return self._bounded_load[dtype](tile, valid_rows, valid_cols)
        else:
            return self._load_fragment[dtype](tile)

    @always_inline
    def _bounded_load[
        dtype: DType
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
    ) -> SIMD[dtype, 8]:
        """Bounded path: predicated loads, zero-fill for OOB.

        Vectorized 4-wide when all 4 cols in bounds; scalar per-element
        when straddling the column boundary. Zero-filled OOB elements
        contribute nothing to the dot product.
        """
        var row_stride = Self._row_stride(tile)
        var col = Int(self.cb)
        var lo = SIMD[dtype, 4](0)
        var hi = SIMD[dtype, 4](0)

        if Int(self.rb) < valid_rows:
            var off = Int(self.rb) * row_stride + col
            if col + 3 < valid_cols:
                lo = (tile.ptr + off).load[width=4]()
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        lo[i] = tile.ptr[off + i]

        if Int(self.rb) + 8 < valid_rows:
            var off = (Int(self.rb) + 8) * row_stride + col
            if col + 3 < valid_cols:
                hi = (tile.ptr + off).load[width=4]()
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        hi[i] = tile.ptr[off + i]

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
        var col = Int(self.cb)

        if Int(self.rb) < valid_rows:
            var off = Int(self.rb) * row_stride + col
            if col + 3 < valid_cols:
                (tile.ptr + off).store(frag.slice[4, offset=0]())
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        tile.ptr[off + i] = frag[i]

        if Int(self.rb) + 8 < valid_rows:
            var off = (Int(self.rb) + 8) * row_stride + col
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

        comptime for ki in range(num_k_steps):
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
                )

                comptime for ni in range(Self.num_n_mmas):
                    var b_sub = b_tile.tile[16, 16](
                        ni if Self.transpose_b else ki,
                        ki if Self.transpose_b else ni,
                    )
                    var b_frag = self._do_load[Self.b_type, bounded](
                        b_sub,
                        (b_valid_cols - ni * 16) if Self.transpose_b else (
                            k_valid - ki * 16
                        ),
                        (k_valid - ki * 16) if Self.transpose_b else (
                            b_valid_cols - ni * 16
                        ),
                    )

                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        a_frag,
                        b_frag,
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
        """Store accumulators with bounds checking.

        Only writes elements where (row < valid_rows) and (col < valid_cols).
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
