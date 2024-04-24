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
)
from memory import stack_allocation
from memory.unsafe import DTypePointer
from math import fma
from .Matmul_v2 import InnerMatmulKernel
from .MatmulLoadStore import (
    _initialize_c_tile_default,
    _load_c_tile_default,
    _store_c_tile_default,
)
from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the default microkernel.
@value
struct Inner_matmul_default[
    config: MatmulConfig,
](InnerMatmulKernel):
    # Parameters for global reference.

    fn _initialize_c_tile[
        a_row_size: Int,
        pack_inner_size: Int,
    ](self, c0_local: NDBuffer,):
        _initialize_c_tile_default[a_row_size, pack_inner_size](c0_local)

    @always_inline
    fn _load_c_tile[
        a_row_size: Int, pack_inner_size: Int, skip_boundary_check: Bool
    ](
        self,
        c_ptr: DTypePointer[config.c_type],
        c_stride: Int,
        c0_local: NDBuffer,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        _load_c_tile_default[a_row_size, pack_inner_size, skip_boundary_check](
            c_ptr, c_stride, c0_local, tile_n_idx, c_bound
        )

    @always_inline
    fn _store_c_tile[
        a_row_size: Int, pack_inner_size: Int, skip_boundary_check: Bool
    ](
        self,
        c_ptr: DTypePointer[config.c_type],
        c_stride: Int,
        c0_local: NDBuffer,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        _store_c_tile_default[a_row_size, pack_inner_size, skip_boundary_check](
            c_ptr, c_stride, c0_local, tile_n_idx, c_bound
        )

    fn _accumulate_[
        a_row_size: Int, pack_inner_size: Int
    ](
        self,
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b_packed: NDBuffer[config.b_type, 3, config.packed_shape],
        c0_local: NDBuffer,
        global_offset: GemmShape,
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            a: TODO.
            b_packed: TODO.
            c0_local: Pre-allocated local buffer for c partial sums.
            global_offset: TODO.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = global_offset.K + tile_n_k_idx[1]

        var b_ptr = b_packed._offset(Index(n_outer_idx, tile_n_k_idx[1], 0))

        # Prefetch B matrix.
        @parameter
        if config.prefetch_b_distance_k > 0:
            alias prefetch_offset = config.prefetch_b_distance_k * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // simd_size):
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # This inner kernels works with non-transposed A.
        var K = a.dim[1]()
        var a_ptr = a.data.offset(global_offset.M * K + global_k)

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                var c_idx = Index(idx0, idx1 * simd_size)
                var a_val = a_ptr[idx0 * K].cast[config.c_type]()
                alias alignment = alignof[SIMD[config.c_type, simd_size]]()
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )
                var b_val = b_ptr.load[width=simd_size, alignment=alignment](
                    idx1 * simd_size
                ).cast[config.c_type]()
                c_val = fma[config.c_type, simd_size](a_val, b_val, c_val)
                c_local.store[width=simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn __inner_matmul__[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
    ](
        self,
        c0: NDBuffer,
        a0: NDBuffer,
        b0_packed: NDBuffer,
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """

        var a = rebind[NDBuffer[config.a_type, 2, config.a_shape]](a0)

        var c = rebind[NDBuffer[config.c_type, 2, config.c_shape]](c0)

        var c_stride = c.dim[1]()

        var b_packed = rebind[NDBuffer[config.b_type, 3, config.packed_shape]](
            b0_packed
        )

        var c_ptr = c.data.offset(global_offset.M * c_stride + global_offset.N)
        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[
            alignof[SIMD[config.c_type, config.simd_size]]()
        ]()

        for idx_n in range(0, tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if global_offset.K == 0:
                self._initialize_c_tile[a_row_size, pack_inner_size](c_local)
            else:
                self._load_c_tile[
                    a_row_size, pack_inner_size, skip_boundary_check
                ](c_ptr, c_stride, c_local, idx_n, c_bound)

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            for idx_k in range(tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_[
                    a_row_size,
                    pack_inner_size,
                ](a, b_packed, c_local, global_offset, Index(idx_n, idx_k))

            self._store_c_tile[
                a_row_size, pack_inner_size, skip_boundary_check
            ](c_ptr, c_stride, c_local, idx_n, c_bound)
