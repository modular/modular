# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

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
from buffer.list import DimList
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
)
from memory import stack_allocation
from memory.unsafe import DTypePointer, bitcast
from math import align_down
from .neon_intrinsics import _neon_dotprod
from .vnni_intrinsics import dot_i8_to_i32_saturated_x86, dot_i8_to_i32_x86
from .Matmul_v2 import InnerMatmulKernel
from .MatmulLoadStore import (
    _initialize_c_tile_default,
    _load_c_tile_default,
    _store_c_tile_default,
)

from utils.index import Index, StaticIntTuple
from utils.loop import unroll


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the VNNI microkernel.
@value
struct Inner_matmul_vnni[
    config: MatmulConfig,
](InnerMatmulKernel):
    # Parameters for global reference.

    fn _initialize_c_tile[
        a_row_size: Int,
        pack_inner_size: Int,
    ](self, c0_local: NDBuffer):
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
        a_row_size: Int,
        is_tail: Bool,
        pack_inner_size: Int,
    ](
        self,
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b_packed: NDBuffer[config.b_type, 3, config.packed_shape],
        c0_local: NDBuffer,
        global_offset: GemmShape,
        tile_n_k_idx: StaticIntTuple[2],
        tile_n_k: StaticIntTuple[2],
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
            tile_n_k: TODO
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
        var global_k = global_offset.K + kl
        var b_ptr = b_packed._offset(Index(n_outer_idx, kl // 4, 0)).bitcast[
            config.c_type
        ]()

        @parameter
        if not is_tail:
            # Prefetch B matrix.
            @parameter
            if config.prefetch_b_distance_k > 0:
                alias prefetch_offset = config.prefetch_b_distance_k * pack_inner_size

                @unroll
                for idx in range(pack_inner_size // simd_size):
                    b_ptr.offset(prefetch_offset + idx * simd_size).prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ]()

        # This inner kernels works with non-transposed A.
        var K = a.dim[1]()

        var a_local = Buffer[config.a_type, 4 * a_row_size].stack_allocation()
        var a_base_ptr = a.data.offset(global_offset.M * K + global_k)
        var a_ptr = a_local.data if (
            is_tail and not has_avx512f()
        ) else a_base_ptr
        var a_ptr_stride = 4 if (is_tail and not has_avx512f()) else K

        var tail_length = tile_n_k[1] - kl

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
                    c_val = _neon_dotprod(
                        c_val,
                        bitcast[config.a_type, simd_size * 4](a_val2),
                        bitcast[config.b_type, simd_size * 4](b_val),
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
        debug_assert(
            tile_n_k[1] % 0 == 0, "K dimension must be a multiple of 4"
        )

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
            var kl = align_down(tile_n_k[1], 4)
            for idx_k in range(0, kl, 4):
                # accumulate data for this (n, k) index
                self._accumulate_[a_row_size, False, pack_inner_size](
                    a,
                    b_packed,
                    c_local,
                    global_offset,
                    Index(idx_n, idx_k),
                    tile_n_k,
                )
            if kl != tile_n_k[1]:
                self._accumulate_[a_row_size, True, pack_inner_size](
                    a,
                    b_packed,
                    c_local,
                    global_offset,
                    Index(idx_n, kl),
                    tile_n_k,
                )
            self._store_c_tile[
                a_row_size, pack_inner_size, skip_boundary_check
            ](c_ptr, c_stride, c_local, idx_n, c_bound)
