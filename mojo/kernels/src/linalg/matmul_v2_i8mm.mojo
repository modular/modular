# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_up
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
from .neon_intrinsics import _neon_matmul
from .Matmul_v2 import InnerMatmulKernel

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the I8MM microkernel.
@value
struct Inner_matmul_i8mm[
    config: MatmulConfig,
](InnerMatmulKernel):
    # Parameters for global reference.

    fn _initialize_c_tile[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
        single_row_i8mm: Bool = False,
    ](self, c0_local: NDBuffer,):
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
    fn _load_c_tile[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
        single_row_i8mm: Bool = False,
    ](
        self,
        c_ptr: DTypePointer[config.c_type],
        c_stride: Int,
        c0_local: NDBuffer,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

        Args:
            c_ptr: TODO.
            c_stride: TODO.
            c0_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
            c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[config.c_type, simd_size] = 0
            if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
                var t0 = c_ptr_loc.load[width=2](
                    c_stride * (2 * idx0 + 0) + 2 * idx1
                )
                var t1 = c_ptr_loc.load[width=2](
                    c_stride * (2 * idx0 + 1) + 2 * idx1
                ) if not single_row_i8mm else SIMD[config.c_type, 2](0)
                c_data = rebind[SIMD[config.c_type, simd_size]](t0.join(t1))
            elif idx1 * 2 <= c_bound[1]:
                var t0 = partial_simd_load[2](
                    c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                    0,
                    c_bound[1] - tile_n_idx - idx1 * 2,
                    0,
                )
                var t1 = partial_simd_load[2](
                    c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                    0,
                    c_bound[1] - tile_n_idx - idx1 * 2,
                    0,
                ) if not single_row_i8mm else SIMD[config.c_type, 2](0)
                c_data = rebind[SIMD[config.c_type, simd_size]](t0.join(t1))

            # Store data to local buffer.
            c_local.store[width=simd_size](
                Index(idx0, idx1 * simd_size),
                rebind[SIMD[config.c_type, simd_size]](c_data),
            )

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    @always_inline
    fn _store_c_tile[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
        single_row_i8mm: Bool = False,
    ](
        self,
        c_ptr: DTypePointer[config.c_type],
        c_stride: Int,
        c0_local: NDBuffer,
        tile_n_idx: Int,
        c_bound: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

        Args:
            c_ptr: TODO.
            c_stride: TODO.
            c0_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
            c_bound: Boundary of valid output space within the local tile, in (a_row_size, TileN).
        """
        alias simd_size = config.simd_size
        var c_local = rebind[
            NDBuffer[
                config.c_type,
                2,
                DimList(a_row_size, pack_inner_size),
            ]
        ](c0_local)
        var c_ptr_loc = c_ptr.offset(tile_n_idx)

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data = c_local.load[width=simd_size](
                Index(idx0, idx1 * simd_size)
            )
            if skip_boundary_check or (idx1 * 2 + 2 <= c_bound[1] - tile_n_idx):
                c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1).store[
                    width=2
                ](c_data.slice[2]())

                @parameter
                if not single_row_i8mm:
                    c_ptr_loc.offset(
                        c_stride * (2 * idx0 + 1) + 2 * idx1
                    ).store[width=2](c_data.slice[2, offset=2]())
            elif idx1 * 2 <= c_bound[1]:
                partial_simd_store(
                    c_ptr_loc.offset(c_stride * (2 * idx0 + 0) + 2 * idx1),
                    0,
                    c_bound[1] - tile_n_idx - idx1 * 2,
                    c_data.slice[2](),
                )

                @parameter
                if not single_row_i8mm:
                    partial_simd_store(
                        c_ptr_loc.offset(c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2, offset=2](),
                    )

        unroll[body, a_row_size, pack_inner_size // simd_size]()

    fn _accumulate_[
        a_row_size: Int,
        pack_inner_size: Int,
        # Skip the output c space boundary check if True.
        skip_boundary_check: Bool,
        single_row_i8mm: Bool = False,
    ](
        self,
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b_packed: NDBuffer[config.b_type, 3, config.packed_shape],
        c0_local: NDBuffer,
        # 3D global offset within the whole matmul problem space.
        global_offset: GemmShape,
        # Dynamic tiling parameter for this inner loop
        #  in (TileN, TileK).
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            a: TODO.
            b_packed: : TODO.
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

        var n_outer_idx = tile_n_k_idx[0] // (pack_inner_size // 2)
        var kl = tile_n_k_idx[1]
        var b_ptr = b_packed._offset(Index(n_outer_idx, kl // 8, 0))

        # This inner kernels works with non-transposed A.
        var K = a.dim(1)
        var a_ptr = a.data.offset(
            global_offset.M * K + global_offset.K + 2 * kl
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

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                alias alignment = alignof[SIMD[config.c_type, simd_size]]()
                var a_val = a_ptr.load[width = simd_size * 4](2 * idx0 * K)
                var b_val = b_ptr.offset(16 * idx1).load[
                    width = simd_size * 4, alignment=alignment
                ]()
                var c_idx = Index(idx0, 4 * idx1)
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )
                c_val = _neon_matmul(c_val, a_val, b_val)
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
        (a_row_size2, TileN, TileK) tile.
        """

        alias a_row_size2 = a_row_size // 2 if a_row_size != 1 else a_row_size
        alias single_row_i8mm = (a_row_size == 1)

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

        for idx_n in range(0, tile_n_k[0], pack_inner_size // 2):
            if global_offset.K == 0:
                self._initialize_c_tile[
                    a_row_size2,
                    pack_inner_size,
                    # Skip the output c space boundary check if True.
                    skip_boundary_check,
                    single_row_i8mm,
                ](c_local)
            else:
                self._load_c_tile[
                    a_row_size2,
                    pack_inner_size,
                    # Skip the output c space boundary check if True.
                    skip_boundary_check,
                    single_row_i8mm,
                ](c_ptr, c_stride, c_local, idx_n, c_bound)
            var kl = align_up(tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate_[
                    a_row_size2,
                    pack_inner_size,
                    # Skip the output c space boundary check if True.
                    skip_boundary_check,
                    single_row_i8mm,
                ](a, b_packed, c_local, global_offset, Index(idx_n, idx_k))
            self._store_c_tile[
                a_row_size2,
                pack_inner_size,
                # Skip the output c space boundary check if True.
                skip_boundary_check,
                single_row_i8mm,
            ](c_ptr, c_stride, c_local, idx_n, c_bound)
