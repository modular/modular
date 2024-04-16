# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down
from sys.info import (
    alignof,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
)
from sys.intrinsics import PrefetchOptions

from buffer.buffer import (
    Buffer,
    NDBuffer,
    partial_simd_load,
    partial_simd_store,
)
from buffer.list import Dim, DimList
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
)
from memory import stack_allocation
from memory.unsafe import DTypePointer, bitcast
from .neon_intrinsics import _neon_dotprod
from .vnni_intrinsics import dot_i8_to_i32_saturated_x86, dot_i8_to_i32_x86
from .Matmul import InnerMatmulKernel

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the VNNI microkernel.
@value
struct Inner_matmul_vnni[
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
    # if not has_neon() this is 4 by default.
    alias prefetch_b_distance = 4

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
            c_local: pre-allocated local buffer for c partial sums.
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

    fn _accumulate_[
        is_tail: Bool
    ](self, c0_local: NDBuffer, tile_n_k_idx: StaticIntTuple[2],):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.
        Args:
            c_local: Pre-allocated local buffer for c partial sums.
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
        var kl = tile_n_k_idx[1]

        # Global K index.
        var global_k = self.global_offset.K + kl
        var b_ptr = self.b_packed._offset(
            Index(n_outer_idx, kl // 4, 0)
        ).bitcast[config.c_type]()

        @parameter
        if not is_tail:
            # Prefetch B matrix.
            # prefetch_b_distance > 0 (=4 if not has_neon() as in this case)
            alias prefetch_offset = self.prefetch_b_distance * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // simd_size):
                b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # This inner kernels works with non-transposed A.
        var K = self.a.dim(1)

        var a_local = Buffer[config.a_type, 4 * a_row_size].stack_allocation()
        var a_base_ptr = self.a.data.offset(self.global_offset.M * K + global_k)
        var a_ptr = a_local.data if (
            is_tail and not has_avx512f()
        ) else a_base_ptr
        var a_ptr_stride = 4 if (is_tail and not has_avx512f()) else K

        var tail_length = self.tile_n_k[1] - kl

        # pack A if (tile_n_k_idx[1] - kl) is 1, 2, or 3
        @parameter
        if is_tail and not has_avx512f():
            for idx0 in range(a_row_size):
                for idx_k in range(tail_length):
                    a_local[4 * idx0 + idx_k] = a_base_ptr[idx0 * K + idx_k]

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // simd_size):
                # width K bytes or K/4 ints, a_ptr is pointer to ints
                var a_val = bitcast[config.c_type, 1](
                    partial_simd_load[4](
                        a_ptr.offset(idx0 * a_ptr_stride), 0, tail_length, 0
                    )
                ) if (is_tail and has_avx512f()) else a_ptr.offset(
                    idx0 * a_ptr_stride
                ).bitcast[
                    config.c_type
                ]().load()

                alias alignment = alignof[SIMD[config.c_type, simd_size]]()
                var c_idx = Index(idx0, idx1 * simd_size)
                var c_val = c_local.load[width=simd_size, alignment=alignment](
                    c_idx
                )

                var b_val = b_ptr.offset(idx1 * simd_size).load[
                    width=simd_size, alignment=alignment
                ]()

                @parameter
                if has_neon_int8_dotprod():
                    var a_val2 = SIMD[config.c_type, simd_size].splat(a_val)
                    c_val = _neon_dotprod[
                        config.a_type, config.b_type, config.c_type, simd_size
                    ](
                        c_val,
                        bitcast[config.a_type, 16](a_val2),
                        bitcast[config.b_type, 16](b_val),
                    )
                elif config.saturated_vnni:
                    c_val = dot_i8_to_i32_saturated_x86[simd_size](
                        c_val, a_val, b_val
                    )
                else:
                    c_val = dot_i8_to_i32_x86[simd_size](c_val, a_val, b_val)
                c_local.store[width=simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn __inner_matmul__(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        debug_assert(
            self.tile_n_k[1] % 0 == 0, "K dimension must be a multiple of 4"
        )
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
            var kl = align_down(self.tile_n_k[1], 4)
            for idx_k in range(0, kl, 4):
                # accumulate data for this (n, k) index
                self._accumulate_[False](c_local, Index(idx_n, idx_k))
            if kl != self.tile_n_k[1]:
                self._accumulate_[True](c_local, Index(idx_n, kl))
            self._store_c_tile(c_local, idx_n)
