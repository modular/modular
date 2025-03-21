# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import prefetch
from sys.info import alignof
from sys.intrinsics import PrefetchOptions

from buffer.buffer import NDBuffer
from memory import UnsafePointer

from utils.index import Index, IndexList

from .accumulate import _Accumulator
from .matmul import InnerMatmulKernel
from .utils import GemmShape, get_matmul_prefetch_b_distance_k


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the default microkernel.
@value
struct Inner_matmul_default(InnerMatmulKernel):
    @always_inline
    fn _accumulate[
        simd_size: Int, kernel_rows: Int, kernel_cols: Int
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

        # Prefetch B matrix.
        alias prefetch_distance = get_matmul_prefetch_b_distance_k()

        @parameter
        if prefetch_distance > 0:
            alias prefetch_offset = prefetch_distance * kernel_cols

            @parameter
            for idx in range(kernel_cols // simd_size):
                prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ](b_ptr.offset(prefetch_offset + idx * simd_size))

        # This inner kernels works with non-transposed A.
        var K = a.dim[1]()
        var a_ptr = a.data.offset(global_offset.M * K + global_k)

        alias c_type = c_local.type

        # Loop over local accumulator tiles.
        @parameter
        for idx0 in range(kernel_rows):

            @parameter
            for idx1 in range(kernel_cols // simd_size):
                alias alignment = alignof[SIMD[c_type, simd_size]]()

                var a_val = a_ptr[idx0 * K]
                var b_val = b_ptr.load[width=simd_size, alignment=alignment](
                    idx1 * simd_size
                )
                c_local.fma(idx0, idx1, a_val, b_val)

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
                    skip_boundary_check,
                )

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            for idx_k in range(tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate[simd_size](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k),
                )
            acc.store(
                rebind[UnsafePointer[Scalar[c.type]]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
                skip_boundary_check,
            )
