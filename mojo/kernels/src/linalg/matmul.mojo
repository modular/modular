# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from math import align_down, align_up, div_ceil, fma
from sys.info import (
    alignof,
    has_avx2,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
    simdwidthof,
)

from algorithm import sync_parallelize, tile, unswitch, vectorize
from buffer.buffer import (
    Buffer,
    NDBuffer,
)
from buffer.list import Dim, DimList
from .Gemv import gemv
from .MatmulUtils import (
    GemmShape,
    MatmulConfig,
    PartitionHeuristic,
    SubMatmulConfig,
    calculate_tile_n_k,
    dispatch_get_kernel_type,
    elementwise_epilogue_type,
    get_kernel_type,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    get_mm_config,
    InnerKernelID,
    select_inner_kernel,
)
from memory import memset_zero, stack_allocation
from memory.unsafe import DTypePointer, bitcast
from runtime.llcl import Runtime

from .matmul_vnni import Inner_matmul_vnni
from .matmul_i8mm import Inner_matmul_i8mm
from .matmul_neon import Inner_matmul_neon
from .matmul_default import Inner_matmul_default

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

# Define a trait that defines the common functions across all existing
# microkernels:
# - _run_inner_loop_default()
# - _run_inner_loop_vnni()
# - _run_inner_loop_neon()
# - _run_inner_loop_i8mm()


trait InnerMatmulKernel(Copyable):
    fn __inner_matmul__[
        kernel_rows: Int,
        kernel_cols: Int,
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
        ...


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


# Interface method
fn tiledMatmulRun[
    config: MatmulConfig,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None,
    global_tile_shape: GemmShape,
    global_tile_offset: GemmShape,
):
    """Interface function to run tiled matmul on a given sub-tile.

    Args:
        alg: InnerMatmulKernel algorithm for microkernel.
        c: Pre-allocated buffer space for result.
        a: Operand A of the matmul.
        b: Operand B of the mamtul.
        elementwise_epilogue_fn: The elementwise epilogue function.
        global_tile_shape: Tile shape this call will process.
        global_tile_offset: Tile offset on the original buffer.
    """

    var tile_n_k = calculate_tile_n_k[
        a.type, b.type, c.type, config.kernel_cols
    ](global_tile_shape)

    var matmul = TiledMatmul[config, elementwise_epilogue_enabled, kernel_id](
        alg,
        c,
        a,
        b,
        tile_n_k,
        global_tile_offset,
        global_tile_shape,
        BTileGenerator[
            config,
            a.type,
            b.type,
            c.type,
            b.shape,
            config.transpose_b,
            config.b_packed,
        ].get(b, tile_n_k),
        elementwise_epilogue_fn,
    )
    matmul._outer_k_loop()


# Tiled Matmul Implementation.
# TODO: not yet supporting transpose_a
@value
struct TiledMatmul[
    config: MatmulConfig,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    algorithm: InnerMatmulKernel,
]:
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: not yet supporting transpose_a.
    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var alg: algorithm
    var c: NDBuffer[c_type, 2, c_shape]
    var a: NDBuffer[a_type, 2, a_shape]
    var b: NDBuffer[b_type, 2, b_shape]
    # Dynamic tile parameter.
    var tile_n_k: StaticIntTuple[2]

    # Tile starting points on the (M,N,K) coordinates.
    var global_tile_offset: GemmShape

    # Tile sizes this routine will process on the (M,N,K) coordinates.
    var global_tile_shape: GemmShape

    var b_tile_generator: BTileGenerator[
        config,
        a_type,
        b_type,
        c_type,
        b_shape,
        config.transpose_b,
        config.b_packed,
    ]

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None

    fn _outer_m_loop[
        last_n_tile: Bool,
        last_k_tile: Bool,
        m_loop_kernel_cols: Int,
    ](self, global_offset: GemmShape, sub_tile_n: Int, sub_tile_k: Int):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

        Parameters:
            last_n_tile: The last n tile.
            last_k_tile: The last k tile.
            m_loop_kernel_cols: Inner dimension of the packed data layout.

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
                m_loop_kernel_cols
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
                self.alg.__inner_matmul__[
                    tile_size,
                    m_loop_kernel_cols,
                    skip_col_bound,
                ](
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
            if kernel_id == InnerKernelID.I8MM:
                tile[
                    row_iteration,
                    VariadicList[Int](2 * config.kernel_rows, 8, 6, 4, 2, 1),
                ](
                    0,  # starting row offset
                    knm_bounds.M,  # row bound
                )
            else:
                tile[
                    row_iteration,
                    VariadicList[Int](config.kernel_rows, 4, 3, 2, 1),
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
                config.kernel_cols, 2 * config.simd_size, config.simd_size
            )
            var primary_tiles = VariadicList[Int](
                tile_n, 2 * config.simd_size, config.simd_size
            )
            tile[secondary_tiles, config.simd_size, m_loop](
                0, valid_col_count, primary_tiles, config.simd_size
            )
        else:
            alias secondary_tiles_packed_b = VariadicList[Int](
                config.kernel_cols
            )
            var primary_tiles_packed_b = VariadicList[Int](tile_n)
            tile[secondary_tiles_packed_b, config.kernel_cols, m_loop](
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
    #  need to remap every time K and kernel_cols changes.
    fn _view_buffer_as(
        self,
        b_packed_ptr: DTypePointer[b_type],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[b_type, 3, config.packed_shape]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

        Args:
            b_packed_ptr: B matrix in packed layout.
            tile_n: Dynamic tile size to use on N dimension.
            tile_k: Dynamic tile size to use on K dimension.
            n_inner_size: Inner dimension size to use for the packed data
                layout.
        """
        return NDBuffer[b_type, 3, config.packed_shape](
            b_packed_ptr.address,
            DimList(tile_n // n_inner_size, tile_k, n_inner_size),
        )


@always_inline
fn _small_matmul[
    transpose_b: Bool,
    epilogue_wrapper: Optional[elementwise_epilogue_type],
](a: NDBuffer[_, 2, _], b: NDBuffer[_, 2, _], c: NDBuffer[_, 2, _],):
    alias simd_width = simdwidthof[c.type]()

    var M = a.dim[0]()
    var N = b.dim[0]() if transpose_b else b.dim[1]()
    var K = a.dim[1]()

    @parameter
    if transpose_b:
        for m in range(M):
            for n in range(N):
                var acc_vector = SIMD[c.type, simd_width]()
                var acc_scalar = Scalar[c.type]()

                @always_inline
                @parameter
                fn compute_fn[width: Int](k: Int):
                    @parameter
                    if width == 1:
                        acc_scalar += (
                            a[m, k].cast[c.type]() * b[n, k].cast[c.type]()
                        )
                    else:
                        acc_vector += (
                            a.load[width=simd_width](m, k).cast[c.type]()
                            * b.load[width=simd_width](n, k).cast[c.type]()
                        )

                vectorize[compute_fn, simd_width, unroll_factor=2](K)

                var val = acc_vector.reduce_add() + acc_scalar

                @parameter
                if epilogue_wrapper:
                    alias func = epilogue_wrapper.value()
                    func[c.type, 1](Index(m, n), val)
                else:
                    c[Index(m, n)] = val
    else:

        @parameter
        @always_inline
        fn normal_update[
            inner_type: DType, width: Int
        ](coords: StaticIntTuple[2], val: SIMD[inner_type, width]):
            c.store[width=width](
                Index(coords[0], coords[1]), rebind[SIMD[c.type, width]](val)
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
                c.store[width=width](coords, rebind[SIMD[c.type, width]](val))

        @always_inline
        @__copy_capture(N)
        @parameter
        fn accum_out_row[
            output_func: fn[type: DType, width: Int] (
                StaticIntTuple[2], SIMD[type, width]
            ) capturing -> None,
        ](m: Int, k: Int):
            var a_val = a[m, k].cast[c.type]()

            @always_inline
            @__copy_capture(a_val)
            @parameter
            fn _wrapper[simd_width: Int](n: Int):
                output_func[c.type, simd_width](
                    Index(m, n),
                    c.load[width=simd_width](m, n)
                    + a_val * b.load[width=simd_width](k, n).cast[c.type](),
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
    single_thread_blocking_override: Bool,
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    @parameter
    if (
        single_thread_blocking_override
        and not config.transpose_a
        and not config.b_packed
        and a.type == b.type
        and b.type == c.type
    ):
        return _small_matmul[
            config.transpose_b,
            elementwise_lambda_fn,
        ](a, b, c)
    constrained[not config.transpose_a, "transpose_a not yet supported"]()

    var shape = GemmShape.get[False, config.transpose_b](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    # Matrix by vector pattern -> use gemv
    if n == 1:
        var out = Buffer[c.type](c.data, c.dim[0]())
        var lhs = a
        var rhs = Buffer[b.type](b.data, b.dim[0]())
        gemv[
            parallelize=True,
            c_size = Dim(),
            c_type = c.type,
            a_shape = a.shape,
            a_type = a.type,
            b_size = Dim(),
            b_type = b.type,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](out, lhs, rhs)
    else:
        var complexity = m * n * k
        var num_tasks = min(
            div_ceil(complexity, get_min_task_size()),
            num_threads if num_threads > 0 else Runtime().parallelism_level(),
        )

        alias use_i8mm = kernel_id == InnerKernelID.I8MM
        alias simd_size = simdwidthof[c.type]()
        alias alignment = alignof[SIMD[c.type, simd_size]]()
        var kh = align_up(k, 8)
        var mh = align_up(m, 2)
        var a_packed_ptr = DTypePointer[a.type]()
        if use_i8mm:
            a_packed_ptr = DTypePointer[a.type].alloc(
                mh * kh, alignment=alignment
            )
        var a_packed = NDBuffer[a.type, 2, a.shape](
            a_packed_ptr, DimList(mh, kh)
        )

        @always_inline
        @__copy_capture(m, k, num_tasks)
        @parameter
        fn pack_task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                a.type,
                b.type,
                c.type,
                PartitionHeuristic.MOJO,
            ](m, 1, k, task_id, num_tasks, kernel_type_m)
            var t0 = sub_matmul_config.offset[0]
            var t1 = t0 + sub_matmul_config.shape[0]
            packA_i8mm[a.type](t0, t1, k, a.data, a_packed_ptr)

        @always_inline
        @__copy_capture(m, k, num_tasks, n, a_packed)
        @parameter
        fn task_func(task_id: Int):
            var sub_matmul_config = get_partitioned_matmul[
                a.type,
                b.type,
                c.type,
                PartitionHeuristic.MOJO,
            ](m, n, k, task_id, num_tasks, kernel_type_m)

            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            alias use_i8mm = kernel_id == InnerKernelID.I8MM

            _submatmul_sequential_sync[
                config, elementwise_lambda_fn, kernel_id
            ](
                alg,
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
](
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    kernel_type_m: Int,
    num_threads: Int = -1,
):
    @parameter
    @always_inline
    fn dispatch_on_kernel_type[kernel_type: Bool]():
        alias config = get_mm_config[
            a_type,
            b_type,
            c_type,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            b_packed=b_packed,
            kernel_type=kernel_type,
        ]()

        alias kernel_id = select_inner_kernel[a.type, b.type, c.type]()

        @parameter
        if kernel_id == InnerKernelID.DEFAULT:
            _matmul_cpu[
                config,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_default(),
                c,
                a,
                b,
                kernel_type_m,
                num_threads,
            )
        elif kernel_id == InnerKernelID.VNNI:
            _matmul_cpu[
                config,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_vnni[saturated_vnni](),
                c,
                a,
                b,
                kernel_type_m,
                num_threads,
            )
        elif kernel_id == InnerKernelID.NEON:
            _matmul_cpu[
                config,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_neon(),
                c,
                a,
                b,
                kernel_type_m,
                num_threads,
            )
        elif kernel_id == InnerKernelID.I8MM:
            _matmul_cpu[
                config,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_i8mm(),
                c,
                a,
                b,
                kernel_type_m,
                num_threads,
            )
        else:
            constrained[False, "no _run_inner_loop implementation"]()

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
    constrained[target == "cpu" or target == "cuda", "unsupported target"]()

    @parameter
    if target == "cpu":
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
        ](c, a, b, kernel_type_m, num_threads)

    else:
        alias config = get_mm_config[
            a_type,
            b_type,
            c_type,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            b_packed=b_packed,
        ]()

        _matmul_gpu[
            config,
            elementwise_lambda_fn,
            single_thread_blocking_override,
        ](
            c,
            a,
            b,
            num_threads,
        )


fn _submatmul_sequential_sync[
    config: MatmulConfig,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
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
                c.type,
                c.shape,
                elementwise_lambda_fn.value(),
            ](
                offset,
                shape,
                c,
            )
        else:
            pass

    tiledMatmulRun[config, elementwise_lambda_fn.__bool__(), kernel_id](
        alg,
        c,
        a,
        b,
        elementwise_closure,
        sub_matrix_shape,
        sub_matrix_offset,
    )


fn _submatmul_sequential_sync[
    config: MatmulConfig,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    saturated_vnni: Bool,
](
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
    kernel_type_m: Int = 0,
):
    alias kernel_id = select_inner_kernel[a.type, b.type, c.type]()

    @parameter
    if kernel_id == InnerKernelID.DEFAULT:
        _submatmul_sequential_sync[config, elementwise_lambda_fn, kernel_id,](
            Inner_matmul_default(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
            kernel_type_m,
        )
    elif kernel_id == InnerKernelID.VNNI:
        _submatmul_sequential_sync[config, elementwise_lambda_fn, kernel_id,](
            Inner_matmul_vnni[saturated_vnni](),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
            kernel_type_m,
        )
    elif kernel_id == InnerKernelID.NEON:
        _submatmul_sequential_sync[config, elementwise_lambda_fn, kernel_id,](
            Inner_matmul_neon(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
            kernel_type_m,
        )
    elif kernel_id == InnerKernelID.I8MM:
        _submatmul_sequential_sync[config, elementwise_lambda_fn, kernel_id,](
            Inner_matmul_i8mm(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
            kernel_type_m,
        )
    else:
        constrained[False, "no _run_inner_loop implementation"]()
