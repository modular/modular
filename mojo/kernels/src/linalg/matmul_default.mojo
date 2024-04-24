# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import fma
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
from .Matmul import InnerMatmulKernel

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the default microkernel.
struct Inner_matmul_default[
    config: MatmulConfig,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
](InnerMatmulKernel):
    # Parameters for global reference.
    var c_stride: Int
    var c_ptr: DTypePointer[config.c_type]
    var a: NDBuffer[config.a_type, 2, config.a_shape]
    var b_packed: NDBuffer[config.b_type, 3, config.packed_shape]
    # 3D global offset within the whole matmul problem space.
    var global_offset: GemmShape
    # Dynamic tiling parameter for this inner loop
    #  in (TileN, TileK).
    var tile_n_k: StaticIntTuple[2]
    # Boundary of valid output space within the
    #  local tile, in (a_row_size, TileN).
    var c_bound: StaticIntTuple[2]

    fn __init__(
        inout self,
        c0: NDBuffer,
        a0: NDBuffer,
        b0_packed: NDBuffer,
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        var a = rebind[NDBuffer[config.a_type, 2, config.a_shape]](a0)
        var b_packed = rebind[NDBuffer[config.b_type, 3, config.packed_shape]](
            b0_packed
        )
        var c = rebind[NDBuffer[config.c_type, 2, config.c_shape]](c0)

        self.c_stride = c.dim[1]()
        self.c_ptr = c.data.offset(
            global_offset.M * self.c_stride + global_offset.N
        )
        self.a = a
        self.b_packed = b_packed
        self.global_offset = global_offset
        self.tile_n_k = tile_n_k
        self.c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

    fn _initialize_c_tile(
        self,
        c0_local: NDBuffer,
    ):
        """Utility function on the inner loop. Initializes a local c buffer with
        all zeros.

        Args:
            c0_local: pre-allocated local buffer for c partial sums.
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)

        @always_inline
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.store[
                width=simd_size,
                alignment = alignof[SIMD[config.c_type, simd_size]](),
            ](
                Index(idx0, idx1 * simd_size),
                SIMD[config.c_type, simd_size](0),
            )

        unroll[outer_body, a_row_size, pack_inner_size // simd_size]()

    @always_inline
    fn _load_c_tile(
        self,
        c0_local: NDBuffer,
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

        Args:
            c0_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)
        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[config.c_type, simd_size] = 0
            if skip_boundary_check or (
                idx1 * simd_size + simd_size <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd load if all within bound
                c_data = c_ptr.load[width=simd_size](idx1 * simd_size)
            elif idx1 * simd_size <= self.c_bound[1]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[simd_size](
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * simd_size,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = 0

            # Store data to local buffer.
            c_local.store(Index(idx0, idx1 * simd_size), c_data)

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    @always_inline
    fn _store_c_tile(
        self,
        c0_local: NDBuffer,
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

        Args:
            c0_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)
        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data = c_local.load[width=simd_size](
                Index(idx0, idx1 * simd_size)
            )
            if skip_boundary_check or (
                idx1 * simd_size + simd_size <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd store if all within bound
                c_ptr.offset(idx1 * simd_size).store[width=simd_size](c_data)
            elif idx1 * simd_size <= self.c_bound[1]:
                # Use partial store if col not in simd bound.
                partial_simd_store(
                    c_ptr.offset(idx1 * simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * simd_size,
                    c_data,
                )

            @parameter
            if idx1 == pack_inner_size // simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    fn _accumulate_default[
        a_col_size: Int
    ](self, c0_local: NDBuffer, tile_n_k_idx: StaticIntTuple[2],):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c0_local: Pre-allocated local buffer for c partial sums.
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

        constrained[a_col_size == 1]()
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, tile_n_k_idx[1], 0)
        )

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
        var K = self.a.dim[1]()
        var a_ptr = self.a.data.offset(self.global_offset.M * K + global_k)

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

    fn __inner_matmul__(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """

        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[
            alignof[SIMD[config.c_type, config.simd_size]]()
        ]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            for idx_k in range(self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_default[1](c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, idx_n)
