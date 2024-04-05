# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import fma, min
from sys.info import alignof

from buffer.buffer import NDBuffer
from buffer.list import DimList
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
)
from memory import stack_allocation
from memory.unsafe import DTypePointer
from .Matmul_v2 import InnerMatmulKernel, LoadStoreOutputTile

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the Neon microkernel.
struct Inner_matmul_neon[
    config: MatmulConfig,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
    single_row_i8mm: Bool = False,
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

        self._initialize_c_tile(c_local)
        return LoadStoreOutputTile[
            config.c_type, simd_size, a_row_size, pack_inner_size, True
        ].run(
            c_local,
            c_ptr,
            self.c_stride,
            min(self.c_bound[1] - tile_n_idx, pack_inner_size),
        )

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

        return LoadStoreOutputTile[
            config.c_type, simd_size, a_row_size, pack_inner_size, False
        ].run(
            c_local,
            c_ptr,
            self.c_stride,
            min(self.c_bound[1] - tile_n_idx, pack_inner_size),
        )

    fn _accumulate_lane[
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

        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // pack_inner_size

        # Global K index.
        var global_k = self.global_offset.K + tile_n_k_idx[1]

        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, tile_n_k_idx[1], 0)
        )

        var a_vals = stack_allocation[
            a_row_size, SIMD[config.c_type, a_col_size]
        ]()

        @unroll
        for row in range(a_row_size):
            var global_m = self.global_offset.M + row
            var a_val = self.a.load[width=a_col_size](global_m, global_k).cast[
                config.c_type
            ]()
            a_vals[row] = a_val

        @unroll
        for lane in range(a_col_size):

            @unroll
            for col in range(pack_inner_size // simd_size):
                var b_val = b_ptr.offset(col * simd_size).load[
                    width=simd_size
                ]().cast[config.c_type]()

                @unroll
                for row in range(a_row_size):
                    var a_val = a_vals[row]
                    var c_idx = Index(row, col * simd_size)
                    var c_val = c_local.load[width=simd_size](c_idx)
                    c_val = fma[config.c_type, simd_size](
                        a_val[lane], b_val, c_val
                    )
                    c_local.store[width=simd_size](c_idx, c_val)

            b_ptr = b_ptr.offset(pack_inner_size)

    fn __inner_matmul__(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        alias simd_size = config.simd_size
        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[alignof[SIMD[config.c_type, simd_size]]()]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)

            var partition_end = simd_size * (self.tile_n_k[1] // simd_size)
            for idx_k0 in range(0, partition_end, simd_size):
                self._accumulate_lane[simd_size](c_local, Index(idx_n, idx_k0))

            for idx_k1 in range(partition_end, self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_lane[1](c_local, Index(idx_n, idx_k1))

            self._store_c_tile(c_local, idx_n)
