# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import align_down, align_up, div_ceil, fma, min
from sys.info import (
    alignof,
    has_avx2,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
    simdwidthof,
)
from sys.intrinsics import PrefetchOptions

from algorithm import sync_parallelize, tile, unswitch, vectorize
from algorithm.functional import tile_and_unswitch
from buffer.buffer import (
    Buffer,
    NDBuffer,
    partial_simd_load,
    partial_simd_store,
)
from buffer.list import Dim, DimList
from .Gemv import gemv
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
    PartitionHeuristic,
    SubMatmulConfig,
    _get_tile_n_k,
    calculate_tile_n_k,
    dispatch_get_kernel_type,
    elementwise_epilogue_type,
    get_kernel_type,
    get_matmul_arch_factor,
    get_min_task_size,
    get_packB_unroll_factor,
    get_partitioned_matmul,
    packA_i8mm,
    get_mm_config,
    use_i8mm_fn,
    use_vnni_fn,
)
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer, bitcast
from .neon_intrinsics import _neon_dotprod, _neon_matmul
from runtime.llcl import Runtime
from .transpose import transpose_inplace
from .vnni_intrinsics import dot_i8_to_i32_saturated_x86, dot_i8_to_i32_x86

from collections import OptionalReg as Optional
from utils.index import Index, StaticIntTuple
from utils.loop import unroll
from utils.static_tuple import StaticTuple

from .MatmulGPU import _matmul_gpu
from .MatmulPack import (
    BTileGenerator,
    PackMatrixCols,
    PackMatrixRows,
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)


fn elementwise_epilogue_c_tile[
    simd_width: Int,
    type: DType,
    c_shape: DimList,
    func: fn[type: DType, width: Int] (
        StaticIntTuple[2], SIMD[type, width]
    ) capturing -> None,
](offset: GemmShape, tile_len: GemmShape, c: NDBuffer[type, 2, c_shape]):
    @always_inline
    @parameter
    fn activation_on_col_chunk[col_chunk_size: Int](idx_n: Int):
        var n_coord = idx_n + offset.N
        for idx_m in range(tile_len.M):
            var m_coord = idx_m + offset.M
            var c_coord = (m_coord, n_coord)
            var c_val = c.load[width=col_chunk_size](c_coord)
            func[type, col_chunk_size](c_coord, c_val)

    vectorize[activation_on_col_chunk, simd_width](tile_len.N)


struct LoadStoreOutputTile[
    type: DType,
    simd_size: Int,
    tile_rows: Int,
    tile_columns: Int,
    is_load: Bool,
]:
    var output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)]
    var row_ptrs: Pointer[DTypePointer[type]]
    var load_store_count: Int

    @always_inline
    fn __init__(
        inout self,
        output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)],
        row_ptrs: Pointer[DTypePointer[type]],
        load_store_count: Int,
    ):
        self.output_tile = output_tile
        self.row_ptrs = row_ptrs
        self.load_store_count = load_store_count

    @always_inline
    fn _load_store_columns[
        base_column: Int,
        column_count: Int,
    ](self):
        """Loads or stores one or more columns from the base column for each
        row of the tile."""

        @unroll
        for row in range(tile_rows):
            # Iterate twice for a pairwise load/store or once for any other access.
            alias column_step = min(column_count, simd_size)

            @unroll
            for col in range(
                base_column, base_column + column_count, column_step
            ):

                @parameter
                if is_load:
                    var data = self.row_ptrs[row].offset(col).load[
                        width=column_step
                    ]()
                    self.output_tile.store(Index(row, col), data)
                else:
                    var data = self.output_tile.load[width=column_step](
                        Index(row, col)
                    )
                    self.row_ptrs[row].offset(col).store(data)

    @always_inline
    fn _load_store_tail[
        base_column: Int,
        tail_size: Int,
    ](self):
        """Loads/stores the last elements of the tile that cannot be accessed
        pairwise."""

        if self.load_store_count & tail_size:
            self._load_store_columns[base_column, tail_size]()

            alias tile_columns_remaining = tile_columns - base_column - tail_size

            @parameter
            if tile_columns_remaining >= tail_size // 2 and tail_size > 1:
                self._load_store_tail[base_column + tail_size, tail_size // 2]()
            return

        @parameter
        if tail_size > 1:
            self._load_store_tail[base_column, tail_size // 2]()

    @always_inline
    fn _load_store_pairwise[
        base_column: Int,
    ](self):
        """Loads/stores all pairwise vectors of the tile and dispatches the
        remaining non-pairwise elements."""

        alias tile_columns_remaining = tile_columns - base_column

        # Support fusion of LDP/STP instructions by emitting pairs of load/store
        # vector instructions.
        @parameter
        if tile_columns_remaining >= 2 * simd_size:
            if self.load_store_count >= base_column + 2 * simd_size:
                self._load_store_columns[base_column, 2 * simd_size]()
                self._load_store_pairwise[base_column + 2 * simd_size]()
                return

        @parameter
        if tile_columns_remaining >= simd_size:
            self._load_store_tail[base_column, simd_size]()

    @staticmethod
    @always_inline
    fn run(
        output_tile: NDBuffer[type, 2, DimList(tile_rows, tile_columns)],
        ptr: DTypePointer[type],
        stride: Int,
        load_store_count: Int,
    ):
        """Interface function to run the load/store output tile.
        Args:
            output_tile(NDBuffer): output register tile buffer.
            ptr(DTypePointer): data buffer to use for transferring the tile
                buffer.
            stride(Int): stride to use when stepping through rows of the data
                buffer.
            load_store_count(Int): number of elements to load/store.
        """
        # Compute the pointers to each row of the memory buffer.
        # Note that the compiler produces better code if each pointer is calculated
        # relative to the previous pointer. Using (N * row) causes the compiler to
        # allocate locals to cache the intermediate results.
        var row_ptrs = stack_allocation[tile_rows, DTypePointer[type]]()

        @unroll
        for row in range(tile_rows):
            row_ptrs[row] = ptr if row == 0 else (row_ptrs[row - 1] + stride)

        var instance = Self(
            output_tile,
            row_ptrs,
            load_store_count,
        )
        instance._load_store_pairwise[0]()


struct MatmulInnerLoopBPacked[
    config: MatmulConfig,
    a_row_size: Int,
    pack_inner_size: Int,
    # Skip the output c space boundary check if True.
    skip_boundary_check: Bool,
    single_row_i8mm: Bool = False,
]:
    """Inner loop implementation for mlas-style tiled matmul. Accumulates a
    tile of input defined by (a_row_size, TileN, TileK).
    """

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

    alias use_vnni = use_vnni_fn[config.a_type, config.b_type, config.c_type]()
    alias use_i8mm = use_i8mm_fn[config.a_type, config.b_type, config.c_type]()

    fn __init__(
        inout self,
        c: NDBuffer[config.c_type, 2, config.c_shape],
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b_packed: NDBuffer[config.b_type, 3, config.packed_shape],
        global_offset: GemmShape,
        tile_n_k: StaticIntTuple[2],
        c_bound: StaticIntTuple[2],
    ):
        self.c_stride = c.dim[1]()
        self.c_ptr = c.data.offset(
            global_offset.M * self.c_stride + global_offset.N
        )
        self.a = a
        self.b_packed = b_packed
        self.global_offset = global_offset
        self.tile_n_k = tile_n_k
        self.c_bound = c_bound

    @staticmethod
    fn run(
        c: NDBuffer[config.c_type, 2, config.c_shape],
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b_packed: NDBuffer[config.b_type, 3, config.packed_shape],
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: StaticIntTuple[2],
    ):
        """Interface function to run the packing routine.

        Args:
            c: Pre-allocated buffer space for packed result.
            a: Data buffer operand A.
            b_packed: Data buffer operand B in packed layout.
            global_offset: Offset to use when indexing the original matrix.
            global_bound: Tile upper boundary of the current tile function call.
            tile_n_k: 2D dimension tuple describing the size of the packed tile
                of B.
        """
        var instance = Self(
            c,
            a,
            b_packed,
            global_offset,
            tile_n_k,
            Index(global_bound.M, global_bound.N)
            - Index(global_offset.M, global_offset.N),
        )
        instance._run_inner_loop()

    fn _initialize_c_tile(
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
    ):
        """Utility function on the inner loop. Initializes a local c buffer with
        all zeros.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
        """

        @always_inline
        @parameter
        fn outer_body[idx0: Int, idx1: Int]():
            c_local.store[
                width = config.simd_size,
                alignment = alignof[SIMD[config.c_type, config.simd_size]](),
            ](
                Index(idx0, idx1 * config.simd_size),
                SIMD[config.c_type, config.simd_size](0),
            )

        unroll[outer_body, a_row_size, pack_inner_size // config.simd_size]()

    @always_inline
    fn _load_c_tile(
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Loads a local c_buffer with the
        value stored in the output buffer space, given the indices within the
        tile being processed.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """

        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @parameter
        if self.use_i8mm:

            @always_inline
            @parameter
            fn body_i8mm[idx0: Int, idx1: Int]():
                var c_data: SIMD[config.c_type, config.simd_size] = 0
                if skip_boundary_check or (
                    idx1 * 2 + 2 <= self.c_bound[1] - tile_n_idx
                ):
                    var t0 = c_ptr.load[width=2](
                        self.c_stride * (2 * idx0 + 0) + 2 * idx1
                    )
                    var t1 = c_ptr.load[width=2](
                        self.c_stride * (2 * idx0 + 1) + 2 * idx1
                    ) if not single_row_i8mm else SIMD[config.c_type, 2](0)
                    c_data = rebind[SIMD[config.c_type, config.simd_size]](
                        t0.join(t1)
                    )
                elif idx1 * 2 <= self.c_bound[1]:
                    var t0 = partial_simd_load[2](
                        c_ptr.offset(self.c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    )
                    var t1 = partial_simd_load[2](
                        c_ptr.offset(self.c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    ) if not single_row_i8mm else SIMD[config.c_type, 2](0)
                    c_data = rebind[SIMD[config.c_type, config.simd_size]](
                        t0.join(t1)
                    )

                # Store data to local buffer.
                c_local.store[width = config.simd_size](
                    Index(idx0, idx1 * config.simd_size),
                    rebind[SIMD[config.c_type, config.simd_size]](c_data),
                )

            unroll[body_i8mm, a_row_size, pack_inner_size // config.simd_size]()
            return

        @parameter
        if has_neon():
            self._initialize_c_tile(c_local)
            return LoadStoreOutputTile[
                config.c_type,
                config.simd_size,
                a_row_size,
                pack_inner_size,
                True,
            ].run(
                c_local,
                c_ptr,
                self.c_stride,
                min(self.c_bound[1] - tile_n_idx, pack_inner_size),
            )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data: SIMD[config.c_type, config.simd_size] = 0
            if skip_boundary_check or (
                idx1 * config.simd_size + config.simd_size
                <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd load if all within bound
                c_data = c_ptr.load[width = config.simd_size](
                    idx1 * config.simd_size
                )
            elif idx1 * config.simd_size <= self.c_bound[1]:
                # Use partial load if row inbound but col not
                #  in simd bound.
                c_data = partial_simd_load[config.simd_size](
                    c_ptr.offset(idx1 * config.simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * config.simd_size,
                    0,
                )
            else:
                # Fill zero if row out of bound
                c_data = 0

            # Store data to local buffer.
            c_local.store(Index(idx0, idx1 * config.simd_size), c_data)

            @parameter
            if idx1 == pack_inner_size // config.simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // config.simd_size]()

    @always_inline
    fn _store_c_tile(
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_idx: Int,
    ):
        """Utility function on the inner loop. Stores the value of a local c
        buffer to the corresponding position in the output buffer space.

        Args:
            c_local: pre-allocated local buffer for c partial sums.
            tile_n_idx: n coordinate within the current processing tile.
        """
        var c_ptr = self.c_ptr.offset(tile_n_idx)

        @parameter
        if self.use_i8mm:

            @always_inline
            @parameter
            fn body_i8mm[idx0: Int, idx1: Int]():
                var c_data = c_local.load[width = config.simd_size](
                    Index(idx0, idx1 * config.simd_size)
                )
                if skip_boundary_check or (
                    idx1 * 2 + 2 <= self.c_bound[1] - tile_n_idx
                ):
                    c_ptr.offset(
                        self.c_stride * (2 * idx0 + 0) + 2 * idx1
                    ).store[width=2](c_data.slice[2]())

                    @parameter
                    if not single_row_i8mm:
                        c_ptr.offset(
                            self.c_stride * (2 * idx0 + 1) + 2 * idx1
                        ).store[width=2](c_data.slice[2, offset=2]())
                elif idx1 * 2 <= self.c_bound[1]:
                    partial_simd_store(
                        c_ptr.offset(self.c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        self.c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2](),
                    )

                    @parameter
                    if not single_row_i8mm:
                        partial_simd_store(
                            c_ptr.offset(
                                self.c_stride * (2 * idx0 + 1) + 2 * idx1
                            ),
                            0,
                            self.c_bound[1] - tile_n_idx - idx1 * 2,
                            c_data.slice[2, offset=2](),
                        )

            unroll[body_i8mm, a_row_size, pack_inner_size // config.simd_size]()
            return

        @parameter
        if has_neon():
            return LoadStoreOutputTile[
                config.c_type,
                config.simd_size,
                a_row_size,
                pack_inner_size,
                False,
            ].run(
                c_local,
                c_ptr,
                self.c_stride,
                min(self.c_bound[1] - tile_n_idx, pack_inner_size),
            )

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            var c_data = c_local.load[width = config.simd_size](
                Index(idx0, idx1 * config.simd_size)
            )
            if skip_boundary_check or (
                idx1 * config.simd_size + config.simd_size
                <= self.c_bound[1] - tile_n_idx
            ):
                # Use simd store if all within bound
                c_ptr.offset(idx1 * config.simd_size).store[
                    width = config.simd_size
                ](c_data)
            elif idx1 * config.simd_size <= self.c_bound[1]:
                # Use partial store if col not in simd bound.
                partial_simd_store(
                    c_ptr.offset(idx1 * config.simd_size),
                    0,
                    self.c_bound[1] - tile_n_idx - idx1 * config.simd_size,
                    c_data,
                )

            @parameter
            if idx1 == pack_inner_size // config.simd_size - 1:
                c_ptr = c_ptr.offset(self.c_stride)

        unroll[body, a_row_size, pack_inner_size // config.simd_size]()

    fn _accumulate[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
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
            for idx in range(pack_inner_size // config.simd_size):
                b_ptr.offset(prefetch_offset + idx * config.simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        var K = self.a.dim[1]()
        var a_ptr = self.a.data.offset(self.global_offset.M * K + global_k)

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // config.simd_size):
                var c_idx = Index(idx0, idx1 * config.simd_size)
                var a_val = a_ptr[idx0 * K].cast[config.c_type]()
                alias alignment = alignof[
                    SIMD[config.c_type, config.simd_size]
                ]()
                var c_val = c_local.load[
                    width = config.simd_size, alignment=alignment
                ](c_idx)
                var b_val = b_ptr.load[
                    width = config.simd_size, alignment=alignment
                ](idx1 * config.simd_size).cast[config.c_type]()
                c_val = fma[config.c_type, config.simd_size](
                    a_val, b_val, c_val
                )
                c_local.store[width = config.simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _accumulate_vnni[
        is_tail: Bool
    ](
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
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
            @parameter
            if config.prefetch_b_distance_k > 0:
                alias prefetch_offset = config.prefetch_b_distance_k * pack_inner_size

                @unroll
                for idx in range(pack_inner_size // config.simd_size):
                    b_ptr.offset(
                        prefetch_offset + idx * config.simd_size
                    ).prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
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
                    a_local[4 * idx0 + idx_k] = a_base_ptr.offset(
                        idx0 * K + idx_k
                    ).load()

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // config.simd_size):
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

                alias alignment = alignof[
                    SIMD[config.c_type, config.simd_size]
                ]()
                var c_idx = Index(idx0, idx1 * config.simd_size)
                var c_val = c_local.load[
                    width = config.simd_size, alignment=alignment
                ](c_idx)

                var b_val = b_ptr.offset(idx1 * config.simd_size).load[
                    width = config.simd_size, alignment=alignment
                ]()

                @parameter
                if has_neon_int8_dotprod():
                    var a_val2 = SIMD[config.c_type, config.simd_size].splat(
                        a_val
                    )
                    c_val = _neon_dotprod[
                        config.a_type,
                        config.b_type,
                        config.c_type,
                        config.simd_size,
                    ](
                        c_val,
                        bitcast[config.a_type, 16](a_val2),
                        bitcast[config.b_type, 16](b_val),
                    )
                elif config.saturated_vnni:
                    c_val = dot_i8_to_i32_saturated_x86[config.simd_size](
                        c_val, a_val, b_val
                    )
                else:
                    c_val = dot_i8_to_i32_x86[config.simd_size](
                        c_val, a_val, b_val
                    )
                c_local.store[width = config.simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _run_inner_loop_vnni(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[Self.use_vnni]()
        debug_assert(
            self.tile_n_k[1] % 0 == 0, "K dimension must be a multipel of 4"
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
                self._accumulate_vnni[False](c_local, Index(idx_n, idx_k))
            if kl != self.tile_n_k[1]:
                self._accumulate_vnni[True](c_local, Index(idx_n, kl))
            self._store_c_tile(c_local, idx_n)

    fn _run_inner_loop_default(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[not Self.use_vnni and not has_neon()]()
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
                self._accumulate[1](c_local, Index(idx_n, idx_k))

            self._store_c_tile(c_local, idx_n)

    fn _accumulate_lane[
        a_col_size: Int
    ](
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing `a_col_size` columns of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k)
                coordinates within the current processing tile to index the
                packed B matrix.
        """
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
            for col in range(pack_inner_size // config.simd_size):
                var b_val = b_ptr.offset(col * config.simd_size).load[
                    width = config.simd_size
                ]().cast[config.c_type]()

                @unroll
                for row in range(a_row_size):
                    var a_val = a_vals[row]
                    var c_idx = Index(row, col * config.simd_size)
                    var c_val = c_local.load[width = config.simd_size](c_idx)
                    c_val = fma[config.c_type, config.simd_size](
                        a_val[lane], b_val, c_val
                    )
                    c_local.store[width = config.simd_size](c_idx, c_val)

            b_ptr = b_ptr.offset(pack_inner_size)

    fn _run_inner_loop_neon(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[has_neon() and not Self.use_vnni and not Self.use_i8mm]()
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

            var partition_end = config.simd_size * (
                self.tile_n_k[1] // config.simd_size
            )
            for idx_k0 in range(0, partition_end, config.simd_size):
                self._accumulate_lane[config.simd_size](
                    c_local, Index(idx_n, idx_k0)
                )

            for idx_k1 in range(partition_end, self.tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_lane[1](c_local, Index(idx_n, idx_k1))

            self._store_c_tile(c_local, idx_n)

    fn _accumulate_i8mm(
        self,
        c_local: NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ],
        tile_n_k_idx: StaticIntTuple[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            c_local: Pre-allocated local buffer for c partial sums.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        var n_outer_idx = tile_n_k_idx[0] // (pack_inner_size // 2)
        var kl = tile_n_k_idx[1]
        var b_ptr = self.b_packed._offset(Index(n_outer_idx, kl // 8, 0))

        # This inner kernels works with non-transposed A.
        var K = self.a.dim(1)
        var a_ptr = self.a.data.offset(
            self.global_offset.M * K + self.global_offset.K + 2 * kl
        )

        # Prefetch B matrix.
        @parameter
        if config.prefetch_b_distance_k > 0:
            alias prefetch_offset = config.prefetch_b_distance_k * pack_inner_size

            @unroll
            for idx in range(pack_inner_size // config.simd_size):
                b_ptr.offset(prefetch_offset + idx * config.simd_size).prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ]()

        # Loop over local accumulator tiles.
        @unroll
        for idx0 in range(a_row_size):

            @unroll
            for idx1 in range(pack_inner_size // config.simd_size):
                alias alignment = alignof[
                    SIMD[config.c_type, config.simd_size]
                ]()
                var a_val = a_ptr.load[width=16](2 * idx0 * K)
                var b_val = b_ptr.offset(16 * idx1).load[
                    width=16, alignment=alignment
                ]()
                var c_idx = Index(idx0, 4 * idx1)
                var c_val = c_local.load[
                    width = config.simd_size, alignment=alignment
                ](c_idx)
                c_val = _neon_matmul(c_val, a_val, b_val)
                c_local.store[width = config.simd_size, alignment=alignment](
                    c_idx, c_val
                )

    fn _run_inner_loop_i8mm(self):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (a_row_size, TileN, TileK) tile.
        """
        constrained[Self.use_i8mm]()

        # Allocate accumulation buffer.
        var c_local = NDBuffer[
            config.c_type,
            2,
            DimList(a_row_size, pack_inner_size),
        ].aligned_stack_allocation[
            alignof[SIMD[config.c_type, config.simd_size]]()
        ]()

        for idx_n in range(0, self.tile_n_k[0], pack_inner_size // 2):
            if self.global_offset.K == 0:
                self._initialize_c_tile(c_local)
            else:
                self._load_c_tile(c_local, idx_n)
            var kl = align_up(self.tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate_i8mm(c_local, Index(idx_n, idx_k))
            self._store_c_tile(c_local, idx_n)

    fn _run_inner_loop(self):
        @parameter
        if Self.use_i8mm:
            self._run_inner_loop_i8mm()
        elif has_neon() and not Self.use_vnni and not Self.use_i8mm:
            self._run_inner_loop_neon()
        elif not Self.use_vnni and not has_neon():
            self._run_inner_loop_default()
        elif Self.use_vnni:
            self._run_inner_loop_vnni()
        else:
            constrained[False, "no _run_inner_loop implementation"]()


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
@value
struct TiledMatmul[
    config: MatmulConfig,
    elementwise_epilogue_enabled: Bool,
]:
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: not yet supporting transpose_a.
    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var c: NDBuffer[config.c_type, 2, config.c_shape]
    var a: NDBuffer[config.a_type, 2, config.a_shape]
    var b: NDBuffer[config.b_type, 2, config.b_shape]
    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]

    # Tile starting points on the (M,N,K) coordinates.
    var global_tile_offset: GemmShape

    # Tile sizes this routine will process on the (M,N,K) coordinates.
    var global_tile_shape: GemmShape

    var b_tile_generator: BTileGenerator[
        config, config.b_type, config.transpose_b, config.b_packed
    ]

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None

    # Interface method
    @staticmethod
    fn run(
        c: NDBuffer[config.c_type, 2, config.c_shape],
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b: NDBuffer[config.b_type, 2, config.b_shape],
        elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None,
        global_tile_shape: GemmShape,
        global_tile_offset: GemmShape,
    ):
        """Interface function to run tiled matmul on a given sub-tile.

        Args:
            c: Pre-allocated buffer space for result.
            a: Operand A of the matmul.
            b: Operand B of the mamtul.
            elementwise_epilogue_fn: The elementwise epilogue function.
            global_tile_shape: Tile shape this call will process.
            global_tile_offset: Tile offset on the original buffer.
        """
        alias use_vnni = use_vnni_fn[
            config.a_type, config.b_type, config.c_type
        ]()
        alias use_i8mm = use_i8mm_fn[
            config.a_type, config.b_type, config.c_type
        ]()
        alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

        var tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, factor
        ](global_tile_shape)

        var matmul = TiledMatmul[config, elementwise_epilogue_enabled,](
            c,
            a,
            b,
            tile_n_k,
            global_tile_offset,
            global_tile_shape,
            BTileGenerator[
                config, config.b_type, config.transpose_b, config.b_packed
            ].get(b, tile_n_k),
            elementwise_epilogue_fn,
        )

        matmul._outer_k_loop()

    fn _outer_m_loop[
        last_n_tile: Bool,
        last_k_tile: Bool,
        m_loop_pack_inner_size: Int,
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

        Parameters:
            last_n_tile: The last n tile.
            last_k_tile: The last k tile.
            m_loop_pack_inner_size: Inner dimension of the packed data layout.

        Args:
            global_offset: 3D global offset within the whole
                matmul problem space.
            sub_tile_n: Dynamic tile size to use on N dimension.
            sub_tile_k: Dynamic tile size to use on K dimension.
        """
        # valid distance in each dimension from the current offset to the end of the matrix
        var knm_bounds = (
            self.global_tile_shape + self.global_tile_offset - global_offset
        )

        @__copy_capture(knm_bounds)
        @parameter
        @always_inline
        fn unswitch_residual_n[skip_col_bound: Bool]():
            var b_packed_tile = self.b_tile_generator.get_tile[
                m_loop_pack_inner_size
            ](
                global_offset,
                Index(sub_tile_n, sub_tile_k),
                Index(knm_bounds.N, knm_bounds.K),
            )

            # Launch the MLoop
            # The upper bounds apply to runtime packing. For pre-packing, the
            # tile has been padded to fit (sub_tile_n, sub_tile_k).
            var sub_tile_n_k = Index(
                min(sub_tile_n, knm_bounds.N), min(sub_tile_k, knm_bounds.K)
            )

            @__copy_capture(sub_tile_n_k, b_packed_tile)
            @parameter
            @always_inline
            fn row_iteration[tile_size: Int](row_offset: Int):
                alias tile_size2 = 2 if tile_size == 1 else tile_size
                alias a_row_size = tile_size2 // 2 if config.use_i8mm else tile_size
                MatmulInnerLoopBPacked[
                    config,
                    a_row_size,
                    m_loop_pack_inner_size,
                    skip_col_bound,
                    tile_size == 1,
                ].run(
                    self.c,
                    self.a,
                    b_packed_tile,
                    global_offset + GemmShape(row_offset, 0, 0),
                    self.global_tile_offset + self.global_tile_shape,
                    sub_tile_n_k,
                )

                @parameter
                if elementwise_epilogue_enabled and last_k_tile:
                    self.elementwise_epilogue_fn(
                        global_offset + GemmShape(row_offset, 0, 0),
                        GemmShape {
                            M: tile_size, N: sub_tile_n_k[0], K: sub_tile_n_k[1]
                        },
                    )

            @parameter
            if config.use_i8mm:
                tile[
                    row_iteration,
                    VariadicList[Int](2 * config.a_row_size, 8, 6, 4, 2, 1),
                ](
                    0,  # starting row offset
                    knm_bounds.M,  # row bound
                )
            else:
                tile[
                    row_iteration,
                    VariadicList[Int](config.a_row_size, 4, 3, 2, 1),
                ](
                    0,  # starting row offset
                    knm_bounds.M,  # row bound
                )

        @parameter
        if has_neon():
            # The performance of the skip_col_bound=True path is the same as
            # skip_col_bound=False, so reduce code size and emit only the
            # skip_col_bound=False path.
            unswitch_residual_n[False]()
        else:
            unswitch[unswitch_residual_n](knm_bounds[1] > sub_tile_n)

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop[
        last_k_tile: Bool
    ](self, global_offset: GemmShape, sub_tile_k: Int):
        """Iterate on the N dimension of the whole problem space.

        Args:
            global_offset: 3D global offset within the whole matmul problem
                space.
            sub_tile_k: Dynamic tile size to use on K dimension.
        """
        var valid_col_count: Int = (
            self.global_tile_shape.N
            + self.global_tile_offset.N
            - global_offset.N
        )
        var tile_n: Int = self.tile_n_k[0]

        @parameter
        @always_inline
        fn m_loop[secondary_tile_size: Int](col_idx: Int, tile_size_n: Int):
            @parameter
            @always_inline
            fn m_loop_switch[last_n_tile: Bool]():
                self._outer_m_loop[
                    last_n_tile, last_k_tile, secondary_tile_size
                ](
                    global_offset + GemmShape(0, col_idx, 0),
                    tile_size_n,
                    sub_tile_k,
                )

            unswitch[m_loop_switch](
                self.global_tile_offset.N + col_idx + tile_size_n
                >= self.global_tile_shape.N
            )

        # if b is packed, the packing was performed offline using a single inner
        # size and tile_n.
        @parameter
        if not config.b_packed:
            alias secondary_tiles = VariadicList[Int](
                config.pack_inner_size, 2 * config.simd_size, config.simd_size
            )
            var primary_tiles = VariadicList[Int](
                tile_n, 2 * config.simd_size, config.simd_size
            )
            tile[secondary_tiles, config.simd_size, m_loop](
                0, valid_col_count, primary_tiles, config.simd_size
            )
        else:
            alias secondary_tiles_packed_b = VariadicList[Int](
                config.pack_inner_size
            )
            var primary_tiles_packed_b = VariadicList[Int](tile_n)
            tile[secondary_tiles_packed_b, config.pack_inner_size, m_loop](
                0, valid_col_count, primary_tiles_packed_b, tile_n
            )

    # Iterate over the K dimension of the gemm space.
    fn _outer_k_loop(
        self,
    ):
        """Iterate on the K dimension of the whole problem space."""

        # Each tiled iteration on the k dimension.
        @always_inline
        @parameter
        fn k_iteration(k_offset: Int, k_tile_size: Int):
            @always_inline
            @parameter
            fn outer_n_loop[last_k_tile: Bool]():
                self._outer_n_loop[last_k_tile](
                    GemmShape(0, 0, k_offset) + self.global_tile_offset,
                    k_tile_size,
                )

            unswitch[outer_n_loop](
                k_offset + k_tile_size + self.global_tile_offset.K
                == self.global_tile_shape.K
            )

        tile[k_iteration](
            0,  # k offset
            self.global_tile_shape.K,  # valid K count
            self.tile_n_k[1],  # max tile k size
        )

    # Utility to reshape the dynamic buffer:
    #  need to remap every time K and pack_inner_size changes.
    fn _view_buffer_as(
        self,
        b_packed_ptr: DTypePointer[config.b_type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[config.b_type, 3, config.packed_shape]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

        Args:
            b_packed_ptr: B matrix in packed layout.
            tile_n: Dynamic tile size to use on N dimension.
            tile_k: Dynamic tile size to use on K dimension.
            n_inner_size: Inner dimension size to use for the packed data
                layout.
        """
        return NDBuffer[config.b_type, 3, config.packed_shape](
            b_packed_ptr.address,
            DimList(tile_n // n_inner_size, tile_k, n_inner_size),
        )


@always_inline
fn _small_matmul[
    type: DType,
    a_shape: DimList,
    b_shape: DimList,
    c_shape: DimList,
    transpose_b: Bool,
    epilogue_wrapper: Optional[elementwise_epilogue_type],
](
    a: NDBuffer[type, 2, a_shape],
    b: NDBuffer[type, 2, b_shape],
    c: NDBuffer[type, 2, c_shape],
):
    alias simd_width = simdwidthof[type]()

    var M = a.dim[0]()
    var N = b.dim[0]() if transpose_b else b.dim[1]()
    var K = a.dim[1]()

    @parameter
    if transpose_b:
        for m in range(M):
            for n in range(N):
                var acc_vector = SIMD[type, simd_width]()
                var acc_scalar = Scalar[type]()

                @always_inline
                @parameter
                fn compute_fn[width: Int](k: Int):
                    @parameter
                    if width == 1:
                        acc_scalar += a[m, k] * b[n, k]
                    else:
                        acc_vector += a.load[width=simd_width](m, k) * b.load[
                            width=simd_width
                        ](n, k)

                vectorize[compute_fn, simd_width, unroll_factor=2](K)

                var val = acc_vector.reduce_add() + acc_scalar

                @parameter
                if epilogue_wrapper:
                    alias func = epilogue_wrapper.value()
                    func[type, 1](Index(m, n), val)
                else:
                    c[Index(m, n)] = val
    else:

        @parameter
        @always_inline
        fn normal_update[
            inner_type: DType, width: Int
        ](coords: StaticIntTuple[2], val: SIMD[inner_type, width]):
            c.store[width=width](
                Index(coords[0], coords[1]), rebind[SIMD[type, width]](val)
            )

        @parameter
        @always_inline
        fn last_update[
            _type: DType, width: Int
        ](coords: StaticIntTuple[2], val: SIMD[_type, width]):
            @parameter
            if epilogue_wrapper:
                alias func = epilogue_wrapper.value()
                func[_type, width](coords, val)
            else:
                c.store[width=width](coords, rebind[SIMD[type, width]](val))

        @always_inline
        @__copy_capture(N)
        @parameter
        fn accum_out_row[
            output_func: fn[type: DType, width: Int] (
                StaticIntTuple[2], SIMD[type, width]
            ) capturing -> None,
        ](m: Int, k: Int):
            var a_val = a[m, k]

            @always_inline
            @__copy_capture(a_val)
            @parameter
            fn _wrapper[simd_width: Int](n: Int):
                output_func[type, simd_width](
                    Index(m, n),
                    c.load[width=simd_width](m, n)
                    + a_val * b.load[width=simd_width](k, n),
                )

            vectorize[_wrapper, simd_width, unroll_factor=2](N)

        for m in range(M):
            memset_zero(c.data + m * N, N)
            for k in range(K - 1):
                accum_out_row[normal_update](m, k)
            accum_out_row[last_update](m, K - 1)


@always_inline
fn _matmul_cpu[
    config: MatmulConfig,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[config.c_type, 2, config.c_shape],
    a: NDBuffer[config.a_type, 2, config.a_shape],
    b: NDBuffer[config.b_type, 2, config.b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    @parameter
    if (
        single_thread_blocking_override
        and not config.transpose_a
        and not config.b_packed
        and config.a_type == config.b_type
        and config.b_type == config.c_type
    ):
        return _small_matmul[
            config.a_type,
            config.a_shape,
            config.b_shape,
            config.c_shape,
            config.transpose_b,
            elementwise_lambda_fn,
        ](
            a,
            rebind[NDBuffer[config.a_type, 2, config.b_shape]](b),
            rebind[NDBuffer[config.a_type, 2, config.c_shape]](c),
        )
    constrained[not config.transpose_a, "transpose_a not yet supported"]()

    var shape = GemmShape.get[False, config.transpose_b](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    # Matrix by vector pattern -> use gemv
    if n == 1:
        var out = Buffer[config.c_type](c.data, c.dim[0]())
        var lhs = rebind[NDBuffer[config.a_type, 2, config.a_shape]](a)
        var rhs = Buffer[config.b_type](b.data, b.dim[0]())
        gemv[
            parallelize=True,
            c_size = Dim(),
            c_type = config.c_type,
            a_shape = config.a_shape,
            a_type = config.a_type,
            b_size = Dim(),
            b_type = config.b_type,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](out, lhs, rhs)
    else:
        var complexity = m * n * k
        var num_tasks = min(
            div_ceil(complexity, get_min_task_size()),
            num_threads if num_threads > 0 else Runtime().parallelism_level(),
        )

        alias use_i8mm = use_i8mm_fn[
            config.a_type, config.b_type, config.c_type
        ]()
        alias simd_size = simdwidthof[config.c_type]()
        alias alignment = alignof[SIMD[config.c_type, simd_size]]()
        var kh = align_up(k, 8)
        var mh = align_up(m, 2)
        var a_packed_ptr = DTypePointer[config.a_type]()
        if use_i8mm:
            a_packed_ptr = DTypePointer[config.a_type].alloc(
                mh * kh, alignment=alignment
            )
        var a_packed = NDBuffer[config.a_type, 2, config.a_shape](
            a_packed_ptr, DimList(mh, kh)
        )

        @always_inline
        @__copy_capture(m, k, num_tasks)
        @parameter
        fn pack_task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                config.a_type,
                config.b_type,
                config.c_type,
                PartitionHeuristic.MOJO,
            ](m, 1, k, task_id, num_tasks, kernel_type_m)
            var t0 = sub_matmul_config.offset[0]
            var t1 = t0 + sub_matmul_config.shape[0]
            packA_i8mm[config.a_type](t0, t1, k, a.data, a_packed_ptr)

        @always_inline
        @__copy_capture(m, k, num_tasks, n, a_packed)
        @parameter
        fn task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                config.a_type,
                config.b_type,
                config.c_type,
                PartitionHeuristic.MOJO,
            ](m, n, k, task_id, num_tasks, kernel_type_m)

            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            _submatmul_sequential_sync[config, elementwise_lambda_fn](
                c,
                a_packed if use_i8mm else a,
                b,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
                kernel_type_m,
            )

        # i8mm partition needs to be optimized as a function of m, n and k
        # Also parallelize currently is slower than asyn_parallelize which is depreciated now.
        # See issue 27734
        if use_i8mm:
            sync_parallelize[pack_task_func](num_tasks)

        # TODO (#12624): Closure captures some state on the stack so this needs
        # to be synchronous in order to keep that state alive
        sync_parallelize[task_func](num_tasks)
        a_packed_ptr.free()


@always_inline
fn matmul_M[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    constrained[target == "cpu" or target == "cuda", "unsupported target"]()
    alias func = _matmul_cpu if target == "cpu" else _matmul_gpu

    @parameter
    @always_inline
    fn dispatch_on_kernel_type[kernel_type: Bool]():
        alias config = get_mm_config[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            b_packed=b_packed,
            kernel_type=kernel_type,
            saturated_vnni=saturated_vnni,
        ]()
        func[config, elementwise_lambda_fn, single_thread_blocking_override,](
            rebind[NDBuffer[config.c_type, 2, config.c_shape]](c),
            rebind[NDBuffer[config.a_type, 2, config.a_shape]](a),
            rebind[NDBuffer[config.b_type, 2, config.b_shape]](b),
            kernel_type_m,
            num_threads,
        )

    var shape = GemmShape.get[False, transpose_b](c, a, b)
    var n = shape.N
    var k = shape.K
    dispatch_get_kernel_type[dispatch_on_kernel_type](kernel_type_m, n, k)


@always_inline
fn matmul[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    num_threads: Int = -1,
):
    var kernel_type_m = 0

    @parameter
    if a_shape.at[0]().has_value():
        kernel_type_m = a_shape.at[0]().get()

    matmul_M[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_a,
        transpose_b,
        b_packed,
        elementwise_lambda_fn,
        saturated_vnni,
        single_thread_blocking_override,
        target,
    ](c, a, b, kernel_type_m, num_threads)


fn _submatmul_sequential_sync[
    config: MatmulConfig,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
](
    c: NDBuffer[config.c_type, 2, config.c_shape],
    a: NDBuffer[config.a_type, 2, config.a_shape],
    b: NDBuffer[config.b_type, 2, config.b_shape],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
    kernel_type_m: Int = 0,
):
    constrained[not config.transpose_a, "transpose_a not yet supported"]()

    fn elementwise_closure(offset: GemmShape, shape: GemmShape):
        @parameter
        if elementwise_lambda_fn:
            elementwise_epilogue_c_tile[
                config.simd_size,
                config.c_type,
                config.c_shape,
                elementwise_lambda_fn.value(),
            ](
                offset,
                shape,
                rebind[NDBuffer[config.c_type, 2, config.c_shape]](c),
            )
        else:
            pass

    TiledMatmul[
        config,
        elementwise_lambda_fn.__bool__(),
    ].run(
        c,
        a,
        b,
        elementwise_closure,
        sub_matrix_shape,
        sub_matrix_offset,
    )
