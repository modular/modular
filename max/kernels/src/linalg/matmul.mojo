# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from collections import OptionalReg
from collections.string.string_slice import get_static_string
from math import align_up, ceildiv
from sys.info import alignof, simdwidthof

from algorithm import sync_parallelize, tile, vectorize
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from memory import memset_zero
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList

from .apple_accelerate import apple_gemv, apple_matmul, use_apple_accelerate_lib
from .gemv import gemv
from .matmul_default import Inner_matmul_default
from .matmul_gpu import _matmul_gpu
from .matmul_i8mm import Inner_matmul_i8mm
from .matmul_neon import Inner_matmul_neon
from .matmul_vnni import Inner_matmul_vnni
from .packing import BTileGenerator
from .utils import (
    GemmShape,
    InnerKernelID,
    KernelConfig,
    calculate_tile_n_k,
    dispatch_get_kernel_type,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
    get_kernel_config,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    select_inner_kernel,
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
        ...


fn elementwise_epilogue_c_tile[
    simd_width: Int,
    type: DType,
    origin: MutableOrigin,
    c_shape: DimList,
    func: fn[type: DType, width: Int, *, alignment: Int = 1] (
        IndexList[2], SIMD[type, width]
    ) capturing [_] -> None,
](
    offset: GemmShape,
    tile_len: GemmShape,
    c: NDBuffer[type, 2, origin, c_shape],
):
    @always_inline
    @parameter
    fn activation_on_col_chunk[col_chunk_size: Int](idx_n: Int):
        var n_coord = idx_n + offset.N
        for idx_m in range(tile_len.M):
            var m_coord = idx_m + offset.M
            var c_coord = Index(m_coord, n_coord)
            var c_val = c.load[width=col_chunk_size](c_coord)
            func[type, col_chunk_size](c_coord, c_val)

    vectorize[activation_on_col_chunk, simd_width](tile_len.N)


# Interface method
fn tiled_matmul_run[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    simd_size: Int,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
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

    alias packed_shape = DimList.create_unknown[3]()
    var matmul = TiledMatmul[
        config,
        transpose_b,
        b_packed,
        elementwise_epilogue_enabled,
        kernel_id,
    ](
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
            transpose_b,
            b_packed,
        ].get(b, tile_n_k),
        elementwise_epilogue_fn,
    )
    matmul._outer_k_loop()


# Tiled Matmul Implementation.
@fieldwise_init
struct TiledMatmul[
    a_mut: Bool,
    b_mut: Bool, //,
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    a_type: DType,
    a_shape: DimList,
    a_origin: Origin[a_mut],
    b_type: DType,
    b_shape: DimList,
    b_origin: Origin[b_mut],
    c_type: DType,
    c_shape: DimList,
    c_origin: MutableOrigin,
    algorithm: InnerMatmulKernel,
](Copyable, Movable):
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    TODO: add tag based implementation dispatch.
    TODO: add fusion hooks.
    """

    var alg: algorithm
    var c: NDBuffer[c_type, 2, c_origin, c_shape]
    var a: NDBuffer[a_type, 2, a_origin, a_shape]
    var b: NDBuffer[b_type, 2, b_origin, b_shape]
    # Dynamic tile parameter.
    var tile_n_k: IndexList[2]

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
        transpose_b,
        b_packed,
        b_origin,
    ]

    var elementwise_epilogue_fn: fn (GemmShape, GemmShape) escaping -> None

    fn _outer_m_loop[
        tile_kernel_cols: Int
    ](
        self,
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
        last_k_tile: Bool,
    ):
        """
        Helper function: Pack a subtile of B and iterate through all the rows
            of C.

        Parameters:
            tile_kernel_cols: Inner dimension of the packed data layout.

        Args:
            global_offset: 3D global offset within the whole
                matmul problem space.
            sub_tile_n: Dynamic tile size to use on N dimension.
            sub_tile_k: Dynamic tile size to use on K dimension.
            last_k_tile: The last k tile.
        """
        # valid distance in each dimension from the current offset to the end of the matrix
        var knm_bounds = (
            self.global_tile_shape + self.global_tile_offset - global_offset
        )

        var b_packed_tile = self.b_tile_generator.get_tile[tile_kernel_cols](
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
        fn row_iteration[tile_kernel_rows: Int](row_offset: Int):
            var skip_boundary_check = knm_bounds[1] > sub_tile_n
            self.alg.__inner_matmul__[
                tile_kernel_rows,
                tile_kernel_cols,
                config.simd_size,
            ](
                self.c,
                self.a,
                b_packed_tile,
                global_offset + GemmShape(row_offset, 0, 0),
                self.global_tile_offset + self.global_tile_shape,
                sub_tile_n_k,
                skip_boundary_check,
            )

            if elementwise_epilogue_enabled and last_k_tile:
                self.elementwise_epilogue_fn(
                    global_offset + GemmShape(row_offset, 0, 0),
                    GemmShape(
                        tile_kernel_rows, sub_tile_n_k[0], sub_tile_n_k[1]
                    ),
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
            ](0, knm_bounds.M)

    # Iterate on the N dimension of the gemm space.
    fn _outer_n_loop(
        self, global_offset: GemmShape, sub_tile_k: Int, last_k_tile: Bool
    ):
        """Iterate on the N dimension of the whole problem space.

        Args:
            global_offset: 3D global offset within the whole matmul problem
                space.
            sub_tile_k: Dynamic tile size to use on K dimension.
            last_k_tile: The last k tile.
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
            self._outer_m_loop[secondary_tile_size](
                global_offset + GemmShape(0, col_idx, 0),
                tile_size_n,
                sub_tile_k,
                last_k_tile,
            )

        # if b is packed, the packing was performed offline using a single inner
        # size and tile_n.
        @parameter
        if not b_packed:
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
            var last_k_tile = (
                k_offset + k_tile_size + self.global_tile_offset.K
                == self.global_tile_shape.K
            )
            self._outer_n_loop(
                GemmShape(0, 0, k_offset) + self.global_tile_offset,
                k_tile_size,
                last_k_tile,
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
        b_packed_ptr: UnsafePointer[Scalar[b_type]],
        tile_n: Int,
        tile_k: Int,
        n_inner_size: Int,
    ) -> NDBuffer[b_type, 3, b_packed_ptr.origin, config.packed_shape]:
        """Utility function to use to map the allocated packing workspace into
        an n-dimensional buffer.

        Args:
            b_packed_ptr: B matrix in packed layout.
            tile_n: Dynamic tile size to use on N dimension.
            tile_k: Dynamic tile size to use on K dimension.
            n_inner_size: Inner dimension size to use for the packed data
                layout.
        """
        return NDBuffer[b_type, 3, _, config.packed_shape](
            b_packed_ptr,
            DimList(tile_n // n_inner_size, tile_k, n_inner_size),
        )


@always_inline
fn _small_matmul[
    transpose_b: Bool,
    epilogue_wrapper: OptionalReg[elementwise_epilogue_type],
](
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    c: NDBuffer[mut=True, _, 2, _, _],
):
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
        ](coords: IndexList[2], val: SIMD[inner_type, width]):
            c.store[width=width](
                Index(coords[0], coords[1]), rebind[SIMD[c.type, width]](val)
            )

        @parameter
        @always_inline
        fn last_update[
            _type: DType, width: Int
        ](coords: IndexList[2], val: SIMD[_type, width]):
            @parameter
            if epilogue_wrapper:
                alias func = epilogue_wrapper.value()
                func[_type, width](coords, val)
            else:
                c.store[width=width](coords, rebind[SIMD[c.type, width]](val))

        @always_inline
        @parameter
        fn accum_out_row[
            output_func: fn[type: DType, width: Int] (
                IndexList[2], SIMD[type, width]
            ) capturing [_] -> None,
        ](m: Int, k: Int):
            var a_val = a[m, k].cast[c.type]()

            @always_inline
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
fn _matmul_cpu_impl[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    single_thread_blocking_override: Bool,
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    num_threads: Int = -1,
) raises:
    @parameter
    if (
        single_thread_blocking_override
        and not b_packed
        and a.type == b.type
        and b.type == c.type
    ):
        return _small_matmul[
            transpose_b,
            elementwise_lambda_fn,
        ](a, b, c)

    var shape = GemmShape.get[transpose_b](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    # Matrix by vector pattern -> use gemv
    if n == 1:
        var out = NDBuffer[c.type, 1](c.data, c.dim[0]())
        var lhs = a
        var rhs = NDBuffer[b.type, 1](b.data, b.dim[0]())
        gemv[
            c_size = out.shape.at[0](),
            c_type = out.type,
            a_shape = lhs.shape,
            a_type = lhs.type,
            b_size = rhs.shape.at[0](),
            b_type = rhs.type,
            parallelize=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](out, lhs, rhs)
    else:
        # SGEMM calls for MacOS >= 13.0.0 and a, b, c of type Float32 are
        # directed to the special Apple-specific implementations.
        # apple_matmul handles generic matmuls.
        # apple_gemv handles cases with M=1 (where apple_matmul is suboptimal).
        @parameter
        if use_apple_accelerate_lib[c.type, a.type, b.type]():
            if m == 1:
                apple_gemv[
                    b_packed=b_packed,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b)
            else:
                # if b_packed = True and transpose_b = True:
                #       input is transposed already. We need apple_matmul with
                #       transpose_b=True.
                # if b_packed = True and transpose_b = False:
                #       input is not transposed. Will be transposed by pack function.
                #       We need apple_matmul with transpose_b=True.
                # if b_packed=False and transpose_b = True:
                #       input is transposed already. We need apple_matmul with
                #       transpose_b=True.
                # if b_packed=False and transpose_b = False:
                #       We need apple_matmul with transpose_b=False.
                alias apple_transpose = True if b_packed else transpose_b
                apple_matmul[
                    transpose_b=apple_transpose,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b)
            return

        var complexity = m * n * k
        var num_tasks = min(
            ceildiv(complexity, get_min_task_size()),
            num_threads if num_threads > 0 else parallelism_level(),
        )

        alias use_i8mm = kernel_id == InnerKernelID.I8MM
        alias simd_size = config.simd_size
        alias alignment = alignof[SIMD[c.type, simd_size]]()
        var kh = align_up(k, 8)
        var mh = align_up(m, 2)
        var a_packed_ptr = UnsafePointer[
            Scalar[a.type],
            alignment=alignment,
        ]()
        if use_i8mm:
            a_packed_ptr = UnsafePointer[
                Scalar[a.type],
                alignment=alignment,
            ].alloc(mh * kh)
        var a_packed = NDBuffer[a.type, 2, _, a.shape](
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
                config.kernel_rows,
                config.kernel_cols,
            ](m, 1, k, task_id, num_tasks)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return
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
                config.kernel_rows,
                config.kernel_cols,
            ](m, n, k, task_id, num_tasks)

            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            alias use_i8mm = kernel_id == InnerKernelID.I8MM

            _submatmul_sequential_sync[
                config, transpose_b, b_packed, elementwise_lambda_fn, kernel_id
            ](
                alg,
                c,
                a_packed if use_i8mm else a.origin_cast[
                    True, MutableAnyOrigin
                ](),
                b,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
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
fn _matmul_cpu[
    *,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    kernel_type_m: Int,
    num_threads: Int = -1,
) raises:
    alias kernel_id = select_inner_kernel[a.type, b.type, c.type]()

    @parameter
    @always_inline
    fn dispatch_on_kernel_type[kernel_type: Bool]() raises:
        alias config = get_kernel_config[
            a.type,
            b.type,
            c.type,
            kernel_type=kernel_type,
        ]()

        @parameter
        if kernel_id == InnerKernelID.DEFAULT:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_default(),
                c,
                a,
                b,
                num_threads,
            )
        elif kernel_id == InnerKernelID.VNNI:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_vnni[saturated_vnni](),
                c,
                a,
                b,
                num_threads,
            )
        elif kernel_id == InnerKernelID.NEON:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_neon(),
                c,
                a,
                b,
                num_threads,
            )
        elif kernel_id == InnerKernelID.I8MM:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                single_thread_blocking_override,
                kernel_id,
            ](
                Inner_matmul_i8mm(),
                c,
                a,
                b,
                num_threads,
            )
        else:
            constrained[False, "no _run_inner_loop implementation"]()

    var shape = GemmShape.get[transpose_b](c, a, b)
    var n = shape.N
    var k = shape.K
    dispatch_get_kernel_type[dispatch_on_kernel_type](kernel_type_m, n, k)


@always_inline
fn matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    var cuda_ctx = Optional[DeviceContext]() if is_cpu[
        target
    ]() else ctx.get_device_context()

    return matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        saturated_vnni=saturated_vnni,
        single_thread_blocking_override=single_thread_blocking_override,
        _trace_description=_trace_description,
        target=target,
    ](c, a, b, cuda_ctx)


@always_inline
fn matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    ctx: Optional[DeviceContext],
) raises:
    constrained[is_valid_target[target](), "unsupported target"]()
    constrained[not transpose_a, "transpose_a not yet supported"]()
    debug_assert(
        is_cpu[target]() or Bool(ctx),
        "expected DeviceContext to be provided if target != cpu",
    )

    @always_inline
    @parameter
    fn description_fn() -> String:
        var shape = GemmShape.get[transpose_b](c, a, b)
        # fmt: off
        return String(
            target,
            ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.type),
            ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.type),
            ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.type),
            ";transpose_a=", transpose_a,
            ";transpose_b=", transpose_b,
            ";b_packed=", b_packed,
        )
        # fmt: on

    # TODO(#23049): Pipe info on whether using faster, saturated_vnni is ok
    with Trace[TraceLevel.OP, target=target](
        # Create a string literal so that the event label works with the
        # AsyncRT profiler, whose event labels must be `StaticString`s.
        get_static_string[
            "matmul(",
            _trace_description,
            ")" if _trace_description else "matmul",
        ](),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):

        @parameter
        if is_cpu[target]():
            var kernel_type_m = a.shape.at[0]().or_else(0)

            # The CPU version of matmul doesn't support compute lambda
            # We wrap it around an epilogue lambda instead.
            @parameter
            @always_inline
            fn compute_lambda_wrapper[
                _type: DType, _width: Int, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[_type, _width]):
                @parameter
                if elementwise_compute_lambda_fn:
                    alias compute_lambda = elementwise_compute_lambda_fn.value()
                    var output = compute_lambda(coords, val)
                    c.store[alignment=alignment](
                        coords, rebind[SIMD[c.type, _width]](output)
                    )

            alias elementwise_lambda_wrapper = OptionalReg[
                elementwise_epilogue_type
            ](
                compute_lambda_wrapper
            ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

            _matmul_cpu[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                saturated_vnni=saturated_vnni,
                single_thread_blocking_override=single_thread_blocking_override,
            ](c, a, b, kernel_type_m)

        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                _trace_description=_trace_description,
            ](c, a, b, ctx.value())


fn _submatmul_sequential_sync[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    alias simd_size = config.simd_size

    fn elementwise_closure(offset: GemmShape, shape: GemmShape):
        @parameter
        if elementwise_lambda_fn:
            elementwise_epilogue_c_tile[
                simd_size,
                c.type,
                c.origin,
                c.shape,
                elementwise_lambda_fn.value(),
            ](
                offset,
                shape,
                c,
            )
        else:
            pass

    tiled_matmul_run[
        config,
        transpose_b,
        b_packed,
        simd_size,
        elementwise_lambda_fn.__bool__(),
        kernel_id,
    ](
        alg,
        c,
        a,
        b,
        elementwise_closure,
        sub_matrix_shape,
        sub_matrix_offset,
    )


fn _submatmul_sequential_sync[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    saturated_vnni: Bool,
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    alias kernel_id = select_inner_kernel[a.type, b.type, c.type]()

    @parameter
    if kernel_id == InnerKernelID.DEFAULT:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_default(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.VNNI:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_vnni[saturated_vnni](),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.NEON:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_neon(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.I8MM:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_i8mm(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    else:
        constrained[False, "no _run_inner_loop implementation"]()
