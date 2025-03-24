# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import fma
from sys.info import alignof

from collections import InlineArray
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer

from utils.index import Index, IndexList

from .accumulate import _Accumulator
from .matmul import InnerMatmulKernel
from .utils import GemmShape


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the Neon microkernel.
@value
struct Inner_matmul_neon(InnerMatmulKernel):
    @always_inline
    fn _accumulate_lane[
        simd_size: Int,
        a_col_size: Int,
        kernel_rows: Int,
        kernel_cols: Int,
    ](
        self,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _, _],
        mut c_local: _Accumulator[
            _, kernel_rows, kernel_cols // simd_size, simd_size
        ],
        global_offset: GemmShape,
        tile_n_k_idx: IndexList[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            a: TODO.
            b_packed: TODO.
            c_local: Pre-allocated local buffer for c partial sums.
            global_offset: TODO.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """

        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // kernel_cols

        # Global K index.
        var global_k = global_offset.K + tile_n_k_idx[1]

        var b_ptr = b_packed._offset(Index(n_outer_idx, tile_n_k_idx[1], 0))

        var a_vals = InlineArray[SIMD[c_local.type, a_col_size], kernel_rows](
            uninitialized=True
        )

        @parameter
        for row in range(kernel_rows):
            var global_m = global_offset.M + row
            var a_val = a.load[width=a_col_size](global_m, global_k).cast[
                c_local.type
            ]()
            a_vals[row] = a_val

        @parameter
        for lane in range(a_col_size):

            @parameter
            for col in range(kernel_cols // simd_size):
                var b_val = b_ptr.offset(col * simd_size).load[
                    width=simd_size
                ]().cast[c_local.type]()

                @parameter
                for row in range(kernel_rows):
                    var a_val = a_vals[row]
                    var c_idx = Index(row, col * simd_size)
                    var c_val = c_local[row, col]
                    c_val = fma[c_local.type, simd_size](
                        a_val[lane], b_val, c_val
                    )
                    c_local[row, col] = c_val

            b_ptr = b_ptr.offset(kernel_cols)

    @always_inline
    fn __inner_matmul__[
        kernel_rows: Int,
        kernel_cols: Int,
        simd_size: Int,
    ](
        self,
        c: NDBuffer,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _, _],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: IndexList[2],
        skip_boundary_check: Bool,
    ):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (kernel_rows, TileN, TileK) tile.
        """

        var c_stride = c.dim[1]()

        var c_ptr = c.data.offset(global_offset.M * c_stride + global_offset.N)
        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = _Accumulator[
            c.type, kernel_rows, kernel_cols // simd_size, simd_size
        ]()

        for idx_n in range(0, tile_n_k[0], kernel_cols):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if global_offset.K == 0:
                acc.init(0)
            else:
                acc.load(
                    rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                    c_stride,
                    idx_n,
                    c_bound,
                )

            var partition_end = simd_size * (tile_n_k[1] // simd_size)
            for idx_k0 in range(0, partition_end, simd_size):
                self._accumulate_lane[simd_size, simd_size](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k0),
                )

            for idx_k1 in range(partition_end, tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_lane[simd_size, 1](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k1),
                )
            acc.store(
                rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
            )
