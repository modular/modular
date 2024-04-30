# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.info import alignof
from sys.intrinsics import PrefetchOptions

from buffer.buffer import (
    NDBuffer,
    partial_simd_load,
    partial_simd_store,
)
from buffer.list import DimList
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
    get_matmul_prefetch_b_distance_k,
)
from .MatmulLoadStore import LoadStore_i8mm
from memory import stack_allocation
from memory.unsafe import DTypePointer
from math import align_up
from .neon_intrinsics import _neon_matmul
from .Matmul_v2 import InnerMatmulKernel
from .accumulate import _Accumulator

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the I8MM microkernel.
@value
struct Inner_matmul_i8mm(InnerMatmulKernel):
    # Parameters for global reference.

    @always_inline
    fn _accumulate[
        simd_size: Int, a_row_size: Int, pack_inner_size: Int
    ](
        self,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _],
        inout c_local: _Accumulator[
            _, a_row_size, pack_inner_size // simd_size, simd_size
        ],
        global_offset: GemmShape,
        tile_n_k_idx: StaticIntTuple[2],
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

        var n_outer_idx = tile_n_k_idx[0] // (pack_inner_size // 2)
        var kl = tile_n_k_idx[1]
        var b_ptr = b_packed._offset(Index(n_outer_idx, kl // 8, 0))

        # This inner kernels works with non-transposed A.
        var K = a.dim(1)
        var a_ptr = a.data.offset(
            global_offset.M * K + global_offset.K + 2 * kl
        )

        # Prefetch B matrix.
        alias prefetch_distance = get_matmul_prefetch_b_distance_k()

        @parameter
        if prefetch_distance > 0:
            alias prefetch_offset = prefetch_distance * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // simd_size):
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                alias alignment = alignof[SIMD[c_local.type, simd_size]]()
                var a_val = a_ptr.load[width = simd_size * 4](2 * idx0 * K)
                var b_val = b_ptr.offset(16 * idx1).load[
                    width = simd_size * 4, alignment=alignment
                ]()
                # var c_idx = Index(idx0, 4 * idx1)
                constrained[simd_size == 4]()
                var c_val = c_local[idx0, idx1]
                c_val = _neon_matmul(c_val, a_val, b_val)
                c_local[idx0, idx1] = c_val

    @always_inline
    fn __inner_matmul__[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
    ](
        self,
        c: NDBuffer,
        a: NDBuffer,
        b_packed: NDBuffer[_, 3, _],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size2, TileN, TileK) tile.
        """
        alias simd_size = simdwidthof[c.type]()

        alias a_row_size2 = a_row_size // 2 if a_row_size != 1 else a_row_size
        alias single_row = (a_row_size == 1)

        var c_stride = c.dim[1]()

        var c_ptr = c.data.offset(global_offset.M * c_stride + global_offset.N)

        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = LoadStore_i8mm[
            c.type,
            simd_size,
            skip_boundary_check,
            single_row,
            a_row_size2,
            pack_inner_size,
        ]()

        for idx_n in range(0, tile_n_k[0], pack_inner_size // 2):
            if global_offset.K == 0:
                acc._initialize_c_tile()
            else:
                acc._load_c_tile(
                    rebind[DTypePointer[c.type]](c_ptr),
                    c_stride,
                    idx_n,
                    c_bound,
                )
            var kl = align_up(tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate[simd_size](
                    a,
                    b_packed,
                    acc.output_tile,
                    global_offset,
                    Index(idx_n, idx_k),
                )
            acc._store_c_tile(
                rebind[DTypePointer[c.type]](c_ptr), c_stride, idx_n, c_bound
            )
