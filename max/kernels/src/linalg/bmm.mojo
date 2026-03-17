# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.math import align_up, ceildiv, gcd
from std.sys import align_of, size_of
from std.sys.info import (
    _has_blackwell_tcgen05,
    _is_amd_rdna,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from std.algorithm import elementwise, sync_parallelize, vectorize
from std.algorithm.functional import _get_start_indices_of_nth_subvolume_uint
from std.algorithm.reduction import _reduce_generator
from buffer import Dim, NDBuffer
from buffer.dimlist import DimList
from std.gpu import block_idx, global_idx
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.memory import AddressSpace
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import A100, is_cpu, is_valid_target
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeInt,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout.tma_async import TMATensorTile, create_tensor_tile, create_tma_tile
from layout.tile_layout import Layout as TileLayout
from std.logger import Logger
from std.runtime.asyncrt import DeviceContextPtr, parallelism_level
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from std.gpu.host.info import B200, H100, _is_sm10x_gpu
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from std.memory import memset_zero
from .matmul.cpu.apple_accelerate import (
    apple_batched_matmul,
    use_apple_accelerate_lib,
)
from .matmul.cpu.impl import _submatmul_sequential_sync
from .matmul.gpu import _matmul_gpu
from .matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from .matmul.gpu.amd import gemm_kernel_amd
from .matmul.gpu.sm100.blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel,
)
from .matmul.gpu.sm100_structured.default.dispatch import (
    batched_matmul_dispatch_sm100_bf16,
)
from .utils import GemmShape
from .utils import elementwise_epilogue_type as matmul_elementwise_epilogue_type
from .utils import (
    get_kernel_config,
    get_kernel_type,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    partition_work,
    use_i8mm_fn,
)
from .utils_gpu import MatmulConfig, MatmulKernels

comptime logger = Logger()

comptime elementwise_epilogue_type = fn[
    c_type: DType,
    width: Int,
    rank: Int,
    *,
    alignment: Int = 1,
](
    IndexList[rank],
    SIMD[c_type, width],
) capturing -> None


# Similar to _get_start_indices_of_nth_subvolume but returns only the batch
# dimensions for matmul, skipping the last 2 dimsnions.
@always_inline
def _get_batch_dims[
    rank: Int
](flat_index: Int, shape: IndexList[rank, ...], out res: type_of(shape)):
    res = {}
    var curr_index = flat_index

    comptime for idx in range(rank - 2):
        # Count from the back, skipping last two dims.
        comptime i = rank - idx - 3
        res[i] = curr_index % shape[i]
        curr_index //= shape[i]


# A utility to reshape NDBuffer with rank > 3 to rank-3.
@always_inline
def _reshape_nd_buffer_with_batch_to_3d(
    buffer: NDBuffer[...],
) -> NDBuffer[
    rank=3,
    buffer.type,
    buffer.origin,
    DimList[
        Dim(),
        buffer.shape.get[buffer.rank - 2](),
        buffer.shape.get[buffer.rank - 1](),
    ](),
    address_space=buffer.address_space,
]:
    comptime rank = buffer.rank
    comptime assert rank >= 3, "expecting at least rank-3 NDBuffer"

    var batch_size = 1

    comptime for i in range(rank - 2):
        batch_size *= buffer.dim[i]()

    var matrix_shape = IndexList[3](
        batch_size, buffer.dim[rank - 2](), buffer.dim[rank - 1]()
    )

    comptime static_shape = DimList[
        Dim(),
        buffer.shape.get[buffer.rank - 2](),
        buffer.shape.get[buffer.rank - 1](),
    ]()

    return NDBuffer[
        rank=3,
        buffer.type,
        buffer.origin,
        static_shape,
        address_space=buffer.address_space,
    ](buffer.data.bitcast[Scalar[buffer.type]](), matrix_shape)


@parameter
def _reshape_to_3d[layout: Layout]() -> Layout:
    comptime rank = len(layout.shape)

    # NOTE: need to cast because int tuple returns comptime int
    comptime last = Int(layout.shape[rank - 1])
    comptime second_last = Int(layout.shape[rank - 2])

    return Layout(
        IntTuple(
            UNKNOWN_VALUE,
            second_last,
            last,
        ),
        IntTuple(
            second_last * last if last != UNKNOWN_VALUE
            and second_last != UNKNOWN_VALUE else UNKNOWN_VALUE,
            last,
            1,
        ),
    )


@always_inline
def _slice_types[
    stride_types: Variadic.TypesOfTrait[CoordLike], n_dims: Int
]() -> Variadic.TypesOfTrait[CoordLike]:
    """
    Slice the last n_dims dimensions of the Coord element types.
    """
    comptime rank = Variadic.size(stride_types)
    comptime assert 0 <= rank - n_dims <= Variadic.size(stride_types)
    comptime assert rank <= Variadic.size(stride_types)

    return Variadic.slice_types[stride_types, rank - n_dims]


@always_inline
def _shape_types_to_3d[
    shape_types: Variadic.TypesOfTrait[CoordLike]
]() -> Variadic.TypesOfTrait[CoordLike]:
    """
    Reshape the shape types to 3D. The last two dimensions stay the same. The
    first dimension will be the product of the batch dimensions if all the batch
    dimensions are static, otherwise it's a runtime dimension.
    """
    comptime rank = Variadic.size(shape_types)
    comptime last_two_dims = _slice_types[shape_types, 2]()
    comptime batch_dims = _slice_types[
        Variadic.reverse[*shape_types], rank - 2
    ]()

    comptime _get_first_dim[dtype: DType, *coords: CoordLike] = Variadic.types[
        T=CoordLike, ComptimeInt[Coord[*coords].static_product]
    ] if Coord[*coords].all_dims_known else Variadic.types[
        T=CoordLike, RuntimeInt[dtype]
    ]

    return Variadic.concat_types[
        _get_first_dim[DType.int64, *batch_dims], last_two_dims
    ]


@always_inline
def _reshape_tile_tensor_with_batch_to_3d(
    tensor: TileTensor,
    out result: TileTensor[
        mut=tensor.mut,
        LayoutType=TileLayout[
            _shape_types_to_3d[tensor.LayoutType._shape_types](),
            _slice_types[tensor.LayoutType._stride_types, 3](),
        ],
        dtype=tensor.dtype,
        origin=tensor.origin,
        address_space=tensor.address_space,
        linear_idx_type=tensor.linear_idx_type,
        element_size=tensor.element_size,
    ],
):
    """
    Reshape the TileTensor with batch dimensions to 3D.
    """

    comptime out_shape_types = type_of(result).LayoutType._shape_types
    comptime out_stride_types = type_of(result).LayoutType._stride_types
    comptime rank = tensor.rank
    comptime assert rank >= 3, "expecting at least rank-3 TileTensor"
    var shape = Tuple[*out_shape_types]()
    var strides = Tuple[*out_stride_types]()

    comptime for i in range(3):
        comptime idx = rank - 3 + i

        # copy the stride
        var stride_ptr = UnsafePointer(to=strides[i])
        comptime StrideType = out_stride_types[i]

        comptime if StrideType.is_static_value:
            stride_ptr.init_pointee_copy(
                rebind[StrideType](Idx[StrideType.static_value]())
            )
        else:
            var stride_val = tensor.layout.stride[idx]().value()
            stride_ptr.init_pointee_copy(
                rebind[StrideType](
                    RuntimeInt[StrideType.DTYPE](
                        Scalar[StrideType.DTYPE](stride_val)
                    )
                )
            )

        # copy the shape
        var shape_ptr = UnsafePointer(to=shape[i])
        comptime ShapeType = out_shape_types[i]

        comptime if ShapeType.is_static_value:
            shape_ptr.init_pointee_copy(
                rebind[ShapeType](Idx[ShapeType.static_value]())
            )
        else:
            var shape_val = tensor.layout.shape[idx]().value()

            comptime if i == 0:
                comptime for batch_idx in range(rank - 3):
                    shape_val *= tensor.layout.shape[batch_idx]().value()

            shape_ptr.init_pointee_copy(
                rebind[ShapeType](
                    RuntimeInt[ShapeType.DTYPE](
                        Scalar[ShapeType.DTYPE](shape_val)
                    )
                )
            )

    return type_of(result)(
        tensor.ptr,
        TileLayout[out_shape_types, out_stride_types](
            Coord(shape^), Coord(strides^)
        ),
    )


@always_inline
def _small_batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, rank=rank, c_type, ...],
    a_buf: NDBuffer[rank=rank, a_type, ...],
    b_buf: NDBuffer[rank=rank, b_type, ...],
) raises:
    comptime simd_width = simd_width_of[c_type]()

    # Get the flattened batch.
    var batch_shape = c_buf.get_shape()
    batch_shape[rank - 2] = 1
    batch_shape[rank - 1] = 1
    var B = batch_shape.flattened_length()

    var M = a_buf.dim[rank - 2]()
    var N = b_buf.dim[rank - 1]()
    var K = a_buf.dim[rank - 1]()

    if M == 1 and N == 1:
        for batch in range(B):
            # Get the indices as (B1, B2, ..., BN, 0, 0) where B is
            # each trailing batch dimension.
            var indices = _get_batch_dims[rank](batch, c_buf.get_shape())

            var a_view = NDBuffer[rank=1, a_type](
                a_buf.data + batch * K, Index(K)
            )
            var b_view = NDBuffer[rank=1, b_type](
                b_buf.data + batch * K, Index(K)
            )

            @always_inline
            @__copy_capture(a_view, b_view)
            @parameter
            def input_fn[
                dtype: DType, width: Int, rank: Int
            ](idx: IndexList[rank]) -> SIMD[dtype, width]:
                return (
                    a_view.load[width=width](idx[0]).cast[dtype]()
                    * b_view.load[width=width](idx[0]).cast[dtype]()
                ).cast[dtype]()

            @always_inline
            @parameter
            def output_fn[
                out_type: DType, width: Int, r: Int
            ](i: IndexList[r], value: SIMD[out_type, width]):
                comptime if elementwise_epilogue_fn:
                    comptime func = elementwise_epilogue_fn.value()
                    func[out_type, width, rank](indices.canonicalize(), value)
                else:
                    # This will store only once as it is a 1D reduction.
                    # Just use the original [B, B1,...,BN, 0, 0] indices.
                    c_buf.store[width=width](indices, value.cast[c_type]())

            @always_inline
            @parameter
            def reduce_impl[
                ty: DType, width: Int
            ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
                return v1 + v2

            _reduce_generator[
                input_fn,
                output_fn,
                reduce_impl,
                single_thread_blocking_override=True,
            ](
                a_view.get_shape().canonicalize(),
                init=Scalar[c_type](0),
                reduce_dim=0,
            )

            _ = indices
            _ = a_view
            _ = b_view

    else:
        for batch in range(B):
            # Get the indices as (B1, B2, ..., BN, 0, 0) where B is
            # each trailing batch dimension.
            var indices = _get_batch_dims[rank](batch, c_buf.get_shape())
            var b_buf_index = indices

            memset_zero(c_buf.data + batch * M * N, M * N)
            for m in range(M):
                indices[rank - 2] = m

                for k in range(K):
                    indices[rank - 1] = k
                    b_buf_index[rank - 2] = k

                    var a_val = a_buf[indices]

                    @always_inline
                    def compute_fn[simd_width: Int](n: Int) unified {mut}:
                        indices[rank - 1] = n
                        b_buf_index[rank - 1] = n

                        var b_val = b_buf.load[width=simd_width](b_buf_index)

                        c_buf.store[width=simd_width](
                            indices,
                            c_buf.load[width=simd_width](indices)
                            + a_val.cast[c_type]() * b_val.cast[c_type](),
                        )

                    vectorize[simd_width, unroll_factor=2](N, compute_fn)

            comptime if elementwise_epilogue_fn:
                for m in range(M):
                    indices[rank - 2] = m

                    @always_inline
                    def apply_epilogue[width: Int](n: Int) unified {mut}:
                        indices[rank - 1] = n
                        var val = c_buf.load[width=width](indices)
                        comptime func = elementwise_epilogue_fn.value()
                        func[c_type, width, rank](indices, val)

                    vectorize[simd_width](N, apply_epilogue)

    return


@always_inline
def batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    *,
    transpose_a: Bool,
    transpose_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: NDBuffer[mut=True, rank=rank, c_type, _, _, _],
    a_buf: NDBuffer[mut=False, rank=rank, a_type, _, _, _],
    b_buf: NDBuffer[mut=False, rank=rank, b_type, _, _, _],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """NDBuffer overload of `batched_matmul`. Converts to TileTensor and
    delegates."""
    batched_matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        elementwise_epilogue_fn=elementwise_epilogue_fn,
        saturated_vnni=saturated_vnni,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](TileTensor(c_buf), TileTensor(a_buf), TileTensor(b_buf), context=context)


@always_inline
def _batched_matmul_cpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    *,
    transpose_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
](
    c_buf: NDBuffer[mut=True, rank=rank, c_type, ...],
    a_buf: NDBuffer[rank=rank, a_type, ...],
    b_buf: NDBuffer[rank=rank, b_type, ...],
) raises:
    comptime assert rank < 5, "max rank for batched matmul is currently 4"

    # Batched matmul calls for MacOS >= 13.0.0 and a, b, c of type Float32 are
    # directed to the special Apple-specific implementation.
    comptime if use_apple_accelerate_lib[c_type, a_type, b_type]():
        apple_batched_matmul[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=elementwise_epilogue_fn,
        ](c_buf, a_buf, b_buf)
        return

    # Flatten to 3D Tensor.
    var c = _reshape_nd_buffer_with_batch_to_3d(c_buf)
    var a = _reshape_nd_buffer_with_batch_to_3d(a_buf)
    var b = _reshape_nd_buffer_with_batch_to_3d(b_buf)
    var batch_size: Int = c.dim[0]()

    var m = c.dim[1]()
    var n = c.dim[2]()
    var k = a.dim[2]()
    var num_threads = parallelism_level()
    # Prevent parallelizing tiny matrices, e.x. 1024x4x4x4.
    var max_num_tasks_batch = min(
        ceildiv(m * n * k * batch_size, get_min_task_size()), batch_size
    )
    # Prevent parallelizing matmul with too many threads.
    var max_num_tasks_matmul = get_matmul_num_tasks[
        a_type, b_type, c_type, simd_width_of[c_type](), True
    ](m, n, k, num_threads) if get_kernel_type(
        m, n, k
    ) else get_matmul_num_tasks[
        a_type, b_type, c_type, simd_width_of[c_type](), False
    ](
        m, n, k, num_threads
    )

    # Define temporary variables to hold num_tasks under testing.
    # This is because the closure can't always capture `var` correctly, issue #12167
    var num_tasks_batch_tmp = min(max_num_tasks_batch, num_threads)
    var num_tasks_matmul_tmp = min(
        max_num_tasks_matmul, num_threads // num_tasks_batch_tmp
    )

    # Prioritize partitioning the batch dimension but if there is more than
    # 20% imbalance, we partition more on the matmul.
    # Imbalance ratio is 1 / min_balance_batch_size
    comptime min_balance_batch_size = 5
    var batch_size_per_task = batch_size // num_tasks_batch_tmp
    if (
        batch_size % num_tasks_batch_tmp != 0
        and batch_size_per_task < min_balance_batch_size
    ):
        # In this case, batches are evenly distributed among tasks, and
        # all threads are used unless the matmul is very small.
        num_tasks_batch_tmp = gcd(batch_size, num_threads)
        num_tasks_matmul_tmp = min(
            max_num_tasks_matmul, num_threads // num_tasks_batch_tmp
        )

    var num_tasks_batch = num_tasks_batch_tmp
    var num_tasks_matmul = num_tasks_matmul_tmp
    var num_tasks = num_tasks_batch * num_tasks_matmul

    @always_inline
    @__copy_capture(a, b, c, num_tasks_batch, num_tasks_matmul, m, n, k)
    @parameter
    def task_func(task_id: Int):
        var a_stride_between_batches = a.size() // a.dim[0]()
        var b_stride_between_batches = b.size() // b.dim[0]()
        var c_stride_between_batches = c.size() // c.dim[0]()

        var batch_task_id, matmul_task_id = divmod(task_id, num_tasks_matmul)

        var num_batches = c.dim[0]()
        # Set the granularity to 1 to divide the batches among tasks
        # as even as possible.
        var batch_range = partition_work(
            batch_task_id, num_tasks_batch, num_batches, 1
        )
        var batch_start = batch_range[0]
        var batches_per_task = batch_range[1]

        # Partition the matmul

        for batch in range(batch_start, batch_start + batches_per_task):
            # Get a 2D view of the 3D Tensor.
            var c_view = NDBuffer[rank=2, c_type](
                c.data + batch * c_stride_between_batches,
                IndexList[2](c.dim[1](), c.dim[2]()),
            )
            var a_view = NDBuffer[rank=2, a_type, a.origin, ...](
                a.data + batch * a_stride_between_batches,
                IndexList[2](a.dim[1](), a.dim[2]()),
            )

            comptime config = get_kernel_config[a_type, b_type, c_type]()
            comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
            comptime simd_size = config.simd_size
            comptime alignment = align_of[SIMD[c_type, simd_size]]()
            var kh = align_up(k, 8)
            var mh = align_up(m, 2)

            var b_view = NDBuffer[rank=2, b_type](
                b.data + batch * b_stride_between_batches,
                IndexList[2](b.dim[1](), b.dim[2]()),
            )

            var batch_coords = _get_start_indices_of_nth_subvolume_uint[2](
                UInt(batch), c_buf.get_shape()
            )

            @parameter
            def elementwise_lambda_2d[
                c_type: DType, width: Int, *, alignment: Int = 1
            ](out_coords: IndexList[2], out_val: SIMD[c_type, width]):
                # the caller provided the elementwise epilogue def over the original
                # buffer rank, not the collapsed buffer rank
                # so un-collapse the batch dims here
                comptime if elementwise_epilogue_fn:
                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    comptime func = elementwise_epilogue_fn.value()
                    func[c_type, width, rank](batch_coords, out_val)

            var sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, config.kernel_rows, config.kernel_cols
            ](m, n, k, matmul_task_id, num_tasks_matmul)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            comptime if use_i8mm:
                a_packed_ptr = alloc[Scalar[a_type]](
                    mh * kh, alignment=alignment
                )
                var a_packed = NDBuffer[rank=2, a_type](
                    a_packed_ptr,
                    IndexList[2](mh, kh),
                )
                packA_i8mm[a_type](0, m, k, a_view.data, a_packed_ptr)

                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed=False,
                    elementwise_lambda_fn=Optional[
                        matmul_elementwise_epilogue_type
                    ](
                        elementwise_lambda_2d
                    ) if elementwise_epilogue_fn else None,
                    saturated_vnni=saturated_vnni,
                ](
                    c_view,
                    a_packed,
                    b_view,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )
                a_packed_ptr.free()
            else:
                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed=False,
                    elementwise_lambda_fn=Optional[
                        matmul_elementwise_epilogue_type
                    ](
                        elementwise_lambda_2d
                    ) if elementwise_epilogue_fn else None,
                    saturated_vnni=saturated_vnni,
                ](
                    c_view,
                    a_view,
                    b_view,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )
            _ = batch_coords

    sync_parallelize[task_func](num_tasks)


def naive_batched_matmul_kernel[
    rank: Int,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    CTensorType: TensorLayout,
    ATensorType: TensorLayout,
    BTensorType: TensorLayout,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
](
    c_tensor: TileTensor[c_type, CTensorType, MutAnyOrigin],  # m
    a_tensor: TileTensor[a_type, ATensorType, ImmutAnyOrigin],  # m * k
    b_tensor: TileTensor[b_type, BTensorType, ImmutAnyOrigin],  # 1 * k
    c_buff_nd_shape: IndexList[rank],
) -> None:
    comptime assert (
        c_tensor.rank == 3 and a_tensor.rank == 3 and b_tensor.rank == 3
    ), "expecting rank-3 TileTensor"
    # Provide evidence for flat indexing constraint (for non-nested layouts)
    comptime assert (
        c_tensor.flat_rank == 3
        and a_tensor.flat_rank == 3
        and b_tensor.flat_rank == 3
    )
    var batch_size = UInt(c_tensor.dim(0))
    var m = UInt(c_tensor.dim(1))
    var n = UInt(c_tensor.dim(2))
    var k = UInt(a_tensor.dim(2))

    var x = Int(global_idx.x)
    var y = Int(global_idx.y)
    var z = Int(block_idx.z)

    if UInt(z) >= batch_size or UInt(x) >= n or UInt(y) >= m:
        return
    var val = Scalar[accum_type](0)

    comptime acc_type = Scalar[accum_type]
    for ki in range(k):
        var b_val = b_tensor[z, x, ki] if transpose_b else b_tensor[z, ki, x]
        val += a_tensor[z, y, ki].cast[accum_type]() * b_val.cast[accum_type]()

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        var nd_corrds = _get_start_indices_of_nth_subvolume_uint[2](
            UInt(z), c_buff_nd_shape
        )
        nd_corrds[rank - 1] = x
        nd_corrds[rank - 2] = y
        elementwise_lambda[c_type, 1, rank](nd_corrds, val.cast[c_type]())
    else:
        c_tensor[z, y, x] = val.cast[c_type]()


def batched_matmul_kernel_gpu[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    CTensorType: TensorLayout,
    ATensorType: TensorLayout,
    BTensorType: TensorLayout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tensor: TileTensor[c_type, CTensorType, MutAnyOrigin],  # m
    a_tensor: TileTensor[a_type, ATensorType, ImmutAnyOrigin],  # m * k
    b_tensor: TileTensor[b_type, BTensorType, ImmutAnyOrigin],  # 1 * k
    m: Int,
    n: Int,
    k: Int,
):
    var batch_idx = Int(block_idx.z)
    var a_ptr = a_tensor.ptr + batch_idx * a_tensor.layout.stride[0]().value()
    var b_ptr = b_tensor.ptr + batch_idx * b_tensor.layout.stride[0]().value()
    var c_ptr = c_tensor.ptr + batch_idx * c_tensor.layout.stride[0]().value()

    comptime k_static = a_tensor.static_shape[2]
    comptime n_static = b_tensor.static_shape[1]

    var a = TileTensor(
        a_ptr,
        TileLayout(
            (Idx(m), Idx[a_tensor.static_shape[2]]()),
            Coord[*_slice_types[ATensorType._stride_types, 2]()](),
        ),
    ).to_layout_tensor()
    var b = TileTensor(
        b_ptr,
        TileLayout(
            Coord[*_slice_types[BTensorType._shape_types, 2]()](),
            Coord[*_slice_types[BTensorType._stride_types, 2]()](),
        ),
    ).to_layout_tensor()
    var c = TileTensor(
        c_ptr,
        TileLayout(
            (Idx(m), Idx[c_tensor.static_shape[2]]()),
            Coord[*_slice_types[CTensorType._stride_types, 2]()](),
        ),
    ).to_layout_tensor()

    @parameter
    def elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        comptime if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_coords = IndexList[3](Int(block_idx.z))
            batch_coords[2] = out_coords[1]
            batch_coords[1] = out_coords[0]
            elementwise_epilogue(batch_coords, val)

    comptime if is_nvidia_gpu():
        multistage_gemm_kernel[
            config=config,
            elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)
    elif is_amd_gpu() and not _is_amd_rdna():
        gemm_kernel_amd[
            config=config,
            elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)


@always_inline
def get_shape_index_list[
    rank: Int, dtype: DType, layout: Layout
](tensor: LayoutTensor[dtype, layout, ...]) -> IndexList[rank]:
    var index_list = IndexList[rank](0)

    comptime for i in range(rank):
        index_list[i] = tensor.dim(i)
    return index_list


@always_inline
def _batched_matmul_gpu[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    *,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: TileTensor[mut=True, c_type, ...],
    a_buf: TileTensor[mut=False, a_type, ...],
    b_buf: TileTensor[mut=False, b_type, ...],
    ctx: DeviceContext,
) raises:
    comptime rank = c_buf.rank
    comptime assert rank >= 3, "expecting at least rank-3 TileTensor"
    comptime assert (
        rank == a_buf.rank == b_buf.rank
    ), "all tensors must have the same rank"
    var c_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(c_buf)
    var a_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(a_buf)
    var b_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(b_buf)

    var batch_size = c_tensor_reshaped.dim(0)
    var m = Int(c_tensor_reshaped.dim(1))
    var n = Int(c_tensor_reshaped.dim(2))
    var k = Int(a_tensor_reshaped.dim(2))

    if batch_size == 0 or m == 0 or n == 0 or k == 0:
        return

    comptime has_static_NK = b_tensor_reshaped.LayoutType._shape_types[
        1
    ].is_static_value and b_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value and a_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value and c_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value

    if batch_size == 1:
        with Trace[TraceLevel.OP]("batched_matmul_via_matmul"):
            # If the batch size is 1, then this is just a matmul and we can use the
            # matmul kernel directly.

            # batch_size==1, so flatten (1, X, Y) → (X, Y)
            # by constructing rank-2 TileTensors directly.
            var c_2d = TileTensor(
                c_tensor_reshaped.ptr,
                row_major(Coord(Idx(m), Idx(n))),
            )
            var a_2d = TileTensor(
                a_tensor_reshaped.ptr,
                row_major(Coord(Idx(m), Idx(k))),
            )
            # Use b's actual dims since their order depends on transpose_b.
            var b_2d = TileTensor(
                b_tensor_reshaped.ptr,
                row_major(
                    Coord(
                        Idx(Int(b_tensor_reshaped.dim(1))),
                        Idx(Int(b_tensor_reshaped.dim(2))),
                    )
                ),
            )

            comptime if elementwise_epilogue_fn:
                comptime elementwise_epilogue = elementwise_epilogue_fn.value()

                @parameter
                @__copy_capture(c_buf)
                def elementwise_epilogue_fn_wrapper[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    out_coords: IndexList[2], val: SIMD[dtype, width]
                ) capturing -> None:
                    var batch_coords = IndexList[rank](0)

                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    elementwise_epilogue(batch_coords, val)

                _matmul_gpu[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_epilogue_fn_wrapper,
                ](c_2d, a_2d, b_2d, ctx=ctx)
            else:
                _matmul_gpu[transpose_b=transpose_b](c_2d, a_2d, b_2d, ctx=ctx)

            return

    comptime a_k = a_tensor_reshaped.LayoutType._shape_types[2].static_value
    comptime c_n = c_tensor_reshaped.LayoutType._shape_types[2].static_value

    # SM100 (B200+) batched BF16 matmul dispatch
    comptime use_SM100_kernels = (
        has_nvidia_gpu_accelerator() and _has_blackwell_tcgen05()
    )
    comptime if use_SM100_kernels and has_static_NK and transpose_b:
        comptime bf16_ok = (a_type == b_type == c_type == DType.bfloat16)
        comptime align_ok = (
            c_n * size_of[c_type]() % 16 == 0
            and a_k * size_of[a_type]() % 16 == 0
        )
        comptime if bf16_ok and align_ok:
            batched_matmul_dispatch_sm100_bf16[
                c_type, a_type, b_type, transpose_b
            ](
                c_tensor_reshaped,
                a_tensor_reshaped,
                b_tensor_reshaped,
                ctx,
            )

            comptime if elementwise_epilogue_fn:
                comptime epilogue = elementwise_epilogue_fn.value()
                # SM100+ supports 32B load/store to global memory.
                comptime simd_size = 32 // size_of[c_type]()

                var c_ndbuf = c_tensor_reshaped._to_ndbuffer()

                @parameter
                @__copy_capture(c_ndbuf)
                def epilogue_wrapper[
                    simd_width: Int, rank: Int, alignment: Int = 1
                ](idx: IndexList[rank]):
                    var c_coord = Index(idx[0], idx[1], idx[2])
                    var c_val = c_ndbuf.load[
                        width=simd_width,
                        alignment=alignment * size_of[c_type](),
                    ](c_coord)
                    epilogue[c_type, simd_width, alignment=alignment](
                        c_coord, c_val
                    )

                elementwise[epilogue_wrapper, simd_size, target="gpu"](
                    Index(batch_size, m, n), ctx
                )

            return

    comptime multistage_gemm_cond = (
        c_n % 128 == 0 and a_k % 32 == 0 and a_k >= 128
    )

    comptime use_A100_kernels = (
        has_nvidia_gpu_accelerator()
        and ctx.default_device_info.compute >= A100.compute
    )

    comptime if has_static_NK and use_A100_kernels and multistage_gemm_cond:
        comptime kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

        comptime batched_matmul_type = batched_matmul_kernel_gpu[
            c_tensor_reshaped.dtype,
            a_tensor_reshaped.dtype,
            b_tensor_reshaped.dtype,
            c_tensor_reshaped.LayoutType,
            a_tensor_reshaped.LayoutType,
            b_tensor_reshaped.LayoutType,
            transpose_b,
            kernels.ampere_128x128_4,
            elementwise_epilogue_fn,
        ]

        var grid_dim = kernels.ampere_128x128_4.grid_dim(UInt(m), UInt(n))

        ctx.enqueue_function[batched_matmul_type, batched_matmul_type](
            c_tensor_reshaped,
            a_tensor_reshaped,
            b_tensor_reshaped,
            m,
            n,
            k,
            grid_dim=(grid_dim[0], grid_dim[1], batch_size),
            block_dim=kernels.ampere_128x128_4.block_dim(),
            shared_mem_bytes=kernels.ampere_128x128_4.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(kernels.ampere_128x128_4.shared_mem_usage())
            ),
        )
    elif has_static_NK and has_amd_gpu_accelerator() and transpose_b:

        @always_inline
        @parameter
        def kernel_helper[
            block_m: Int,
            block_n: Int,
            *,
            num_k_partitions: Int = 1,
            num_pipeline_stages: Int = 1,
        ]() raises:
            comptime block_k = 64
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(block_m, block_n, block_k),
                warp_tile_shape=Index(block_m // 2, block_n // 2, block_k),
                num_pipeline_stages=1,
                num_k_partitions=1,
            )

            comptime batched_matmul_type = batched_matmul_kernel_gpu[
                c_tensor_reshaped.dtype,
                a_tensor_reshaped.dtype,
                b_tensor_reshaped.dtype,
                c_tensor_reshaped.LayoutType,
                a_tensor_reshaped.LayoutType,
                b_tensor_reshaped.LayoutType,
                transpose_b,
                config,
                elementwise_epilogue_fn,
            ]

            ctx.enqueue_function[batched_matmul_type, batched_matmul_type](
                c_tensor_reshaped,
                a_tensor_reshaped,
                b_tensor_reshaped,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(n, block_n),
                    ceildiv(m, block_m),
                    batch_size,
                ),
                block_dim=(256, 1, 1),
            )

        # DeepSeek size tuning
        if m == 256 and n == 128 and k == 512:
            kernel_helper[128, 64]()
        elif m == 256 and n == 512 and k == 128:
            kernel_helper[64, 64]()
        elif m == 14 and n == 3072 and k == 12288:
            kernel_helper[32, 32]()
        elif m == 600 and n == 18256 and k == 4096:
            kernel_helper[128, 64]()
        else:
            kernel_helper[128, 128]()

    else:
        c_shape = coord_to_index_list(c_buf.layout.shape_coord())

        comptime BLOCK_DIM = 16
        comptime bmm = naive_batched_matmul_kernel[
            rank,
            c_type,
            a_type,
            b_type,
            c_tensor_reshaped.LayoutType,
            a_tensor_reshaped.LayoutType,
            b_tensor_reshaped.LayoutType,
            transpose_b,
            elementwise_epilogue_fn,
        ]
        ctx.enqueue_function[bmm, bmm](
            c_tensor_reshaped,
            a_tensor_reshaped,
            b_tensor_reshaped,
            c_shape,
            grid_dim=(
                ceildiv(n, BLOCK_DIM),
                ceildiv(m, BLOCK_DIM),
                batch_size,
            ),
            block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
        )


@always_inline
def batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    *,
    transpose_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: NDBuffer[mut=True, rank=rank, c_type, _, _, _],
    a_buf: NDBuffer[mut=False, rank=rank, a_type, _, _, _],
    b_buf: NDBuffer[mut=False, rank=rank, b_type, _, _, _],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """NDBuffer overload of `batched_matmul` (no transpose_a). Converts to
    TileTensor and delegates."""
    batched_matmul[
        transpose_b=transpose_b,
        elementwise_epilogue_fn=elementwise_epilogue_fn,
        saturated_vnni=saturated_vnni,
        target=target,
    ](
        TileTensor(c_buf),
        TileTensor(a_buf),
        TileTensor(b_buf),
        context=context,
    )


@always_inline
def batched_matmul[
    *,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    a_buf: TileTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
    b_buf: TileTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """TileTensor primary implementation of `batched_matmul`."""
    comptime assert c_buf.rank >= 2, "c must be at least rank 2"
    comptime assert (
        c_buf.rank == a_buf.rank == b_buf.rank
    ), "all tensors must have the same rank"
    comptime assert (
        c_buf.flat_rank == c_buf.rank
    ), "c must have a non-nested layout"
    comptime assert (
        a_buf.flat_rank == a_buf.rank
    ), "a must have a non-nested layout"
    comptime assert (
        b_buf.flat_rank == b_buf.rank
    ), "b must have a non-nested layout"
    comptime assert not transpose_a, "transpose_a not yet supported"

    comptime rank = c_buf.rank

    # Construct NDBuffers at call boundary for internal functions.
    comptime dim[i: Int] = Dim(i) if i > -1 else Dim()
    comptime _c_dim[idx: Int]: Dim = dim[c_buf.static_shape[idx]]
    comptime _a_dim[idx: Int]: Dim = dim[a_buf.static_shape[idx]]
    comptime _b_dim[idx: Int]: Dim = dim[b_buf.static_shape[idx]]
    comptime c_shape = DimList[*Variadic.tabulate[rank, _c_dim[_]]]()
    comptime a_shape = DimList[*Variadic.tabulate[rank, _a_dim[_]]]()
    comptime b_shape = DimList[*Variadic.tabulate[rank, _b_dim[_]]]()

    var c_nd = NDBuffer[rank=rank, c_buf.dtype, MutAnyOrigin, c_shape](
        c_buf.ptr.as_any_origin(),
        rebind[IndexList[rank]](
            coord_to_index_list(c_buf.layout.shape_coord())
        ),
    )
    var a_nd = NDBuffer[rank=rank, a_buf.dtype, ImmutAnyOrigin, a_shape](
        a_buf.ptr.as_any_origin(),
        rebind[IndexList[rank]](
            coord_to_index_list(a_buf.layout.shape_coord())
        ),
    )
    var b_nd = NDBuffer[rank=rank, b_buf.dtype, ImmutAnyOrigin, b_shape](
        b_buf.ptr.as_any_origin(),
        rebind[IndexList[rank]](
            coord_to_index_list(b_buf.layout.shape_coord())
        ),
    )

    @always_inline
    @parameter
    def description_fn() -> String:
        # fmt: off
        return String(
            trace_arg("A", a_nd.dynamic_shape, a_nd.dtype),
            ";", trace_arg("B", b_nd.dynamic_shape, b_nd.dtype),
            ";", trace_arg("C", c_nd.dynamic_shape, c_nd.dtype),
            ";transpose_a=", transpose_a,
            ";transpose_b=", transpose_b,
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        "batched_matmul",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        # TODO: generalize to > rank 3
        comptime if (
            single_thread_blocking_override
            and not transpose_b
            and is_cpu[target]()
        ):
            return _small_batched_matmul[
                rank,
                a_buf.dtype,
                b_buf.dtype,
                c_buf.dtype,
                elementwise_epilogue_fn,
            ](
                c_nd.make_dims_unknown(),
                a_nd.make_dims_unknown(),
                b_nd.make_dims_unknown(),
            )

        comptime assert is_valid_target[target](), "unsupported target"

        comptime if is_cpu[target]():
            _batched_matmul_cpu[
                transpose_b=transpose_b,
                elementwise_epilogue_fn=elementwise_epilogue_fn,
                saturated_vnni=saturated_vnni,
            ](
                c_nd.make_dims_unknown(),
                a_nd.make_dims_unknown(),
                b_nd.make_dims_unknown(),
            )
        else:
            comptime assert (
                saturated_vnni == False
            ), "saturated_vnni is not applicable on the gpu"
            _batched_matmul_gpu[
                transpose_b=transpose_b,
                elementwise_epilogue_fn=elementwise_epilogue_fn,
            ](
                c_buf,
                a_buf,
                b_buf,
                context.get_device_context(),
            )


@always_inline
def batched_matmul_shape[
    rank: Int,
    a_type: DType,
    b_type: DType,
    single_thread_blocking_override: Bool,
](
    a_buff: NDBuffer[rank=rank, a_type, ...],
    b_buff: NDBuffer[rank=rank, b_type, ...],
) raises -> IndexList[rank]:
    """
    Compute the output shape of a `batch_matmul` operation, and assert the
    inputs are compatible.

    Parameters:
        rank: Rank of the input and output tensors.
        a_type: Type of the lhs input tensor.
        b_type: Type of the rhs input tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        a_buff: The lhs input tensor.
        b_buff: The rhs input tensor.

    Returns:
        The output shape.
    """

    if rank <= 2:
        raise Error("[batch_matmul] requires rank > 2")

    if a_buff.dim(rank - 1) != b_buff.dim(rank - 2):
        raise Error("[batch_matmul] inputs inner dimensions must match")

    # Check batch dimensions
    var foundMismatch = False

    for i in range(rank - 2):
        if a_buff.dim(i) != b_buff.dim(i):
            foundMismatch = True

    if foundMismatch:
        raise Error("[batch_matmul] inputs batch dimensions must match")

    var output_shape = a_buff.get_shape()
    output_shape[rank - 1] = b_buff.dim(rank - 1)

    return output_shape


comptime _2D_layout[layout: Layout] = Layout(
    IntTuple(layout.shape[1], layout.shape[2]),
    IntTuple(layout.stride[1], layout.stride[2]),
)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
def _bmm_sm100_blockwise_scaled_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c_tensor: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    b_scales_tensor: LayoutTensor[
        b_scales_type, b_scales_layout, ImmutAnyOrigin
    ],
    num_iters: UInt,
):
    comptime c_2d_layout: Layout = _2D_layout[c_layout]
    comptime b_scales_2d_layout: Layout = _2D_layout[b_scales_layout]

    var M = c_tensor.dim(1)
    var N = c_tensor.dim(2)

    var b_scales_ptr = b_scales_tensor.ptr + (
        block_idx.z
        * UInt(b_scales_tensor.dim(1))
        * UInt(b_scales_tensor.dim(2))
    )

    var c = LayoutTensor[c_type, c_2d_layout](
        c_tensor.ptr_at_offset(Index(block_idx.z, 0, 0)),
        RuntimeLayout[c_2d_layout](
            Index(c_tensor.dim(1), c_tensor.dim(2)),
            Index(c_tensor.stride(1), c_tensor.stride(2)),
        ),
    )

    var b_scales = LayoutTensor[b_scales_type, b_scales_2d_layout](
        b_scales_ptr,
        RuntimeLayout[b_scales_2d_layout].row_major(
            IndexList[2](b_scales_tensor.dim(1), b_scales_tensor.dim(2)),
        ),
    )

    @parameter
    def elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        comptime if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_coords = IndexList[3](Int(block_idx.z))
            batch_coords[2] = out_coords[1]
            batch_coords[1] = out_coords[0]
            elementwise_epilogue(batch_coords, val)

    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        a_layout,
        type_of(c).layout,
        a_scales_layout,
        type_of(b_scales).layout,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        block_tile_shape,
        mma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=num_threads,
        elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
            elementwise_epilogue_fn_wrapper
        ) if elementwise_lambda_fn else None,
    ](
        a_tma_op,
        b_tma_op,
        c,
        a_scales_tma_op,
        b_scales,
        num_iters,
    )


def bmm_sm100_blockwise_scaled_fp8[
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, ...],
    a: LayoutTensor[mut=False, a_type, a_layout, ...],
    b: LayoutTensor[mut=False, b_type, b_layout, ...],
    a_scales: LayoutTensor[mut=False, a_scales_type, a_scales_layout, ...],
    b_scales: LayoutTensor[mut=False, b_scales_type, b_scales_layout, ...],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        b_scales_type == a_scales_type == DType.float32
    ), "Only support float32 for a_scales and b_scales"

    comptime assert c.rank == 3, "Only support rank 3 tensors"

    comptime assert (
        c.rank == b.rank and c.rank == a.rank
    ), "all tensors must have the same rank"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime assert BK == 128, "blockwise scaled fp8 only works with BK = 128"

    var batch_size = c.dim(0)
    var M = c.dim(1)
    var N = c.dim(2)
    var K = a.dim(2)

    if batch_size == 0 or M == 0 or N == 0 or K == 0:
        return

    var a_scales_dim0 = a_scales.dim(1)
    var a_scales_dim1 = a_scales.dim(2)
    var b_scales_dim0 = b_scales.dim(1)
    var b_scales_dim1 = b_scales.dim(2)

    if (
        a_scales_dim0 != b_scales_dim1
        or K % a_scales_dim0 != 0
        or (K // a_scales_dim0) != BK
    ):
        raise Error(
            "a_scales_3D.dim(1) must be equal to b_scales.dim(1) and K must be"
            " divisible by a_scales.dim(0) and (K // a_scales.dim(0)) must be"
            " equal to 128"
        )

    if N % b_scales_dim0 != 0 or (N // b_scales_dim0) != BK:
        raise Error(
            "N must be divisible by b_scales.dim(0) and (N // b_scales.dim(0)) "
            " must be equal to 128"
        )

    var padding_size = 16 // size_of[a_scales_type]()
    if a_scales_dim1 % padding_size != 0:
        raise Error(
            "a_scales_3D.dim(2) must be divisible by 16 bytes. This is required"
            " by NVIDIA SM90+ TMA instructions!"
        )

    logger.info(
        "Executing SM100 Basic Batched 1D2D Blockwise Scaled FP8 GEMM"
        " (BLOCK_SCALE_SIZE = 128)"
    )
    logger.info(
        "Problem Shape: MNK=[", batch_size, ", ", M, ", ", N, ", ", K, "]"
    )
    logger.info(
        "A Scales Shape: [",
        a_scales.dim(1),
        ", ",
        a_scales.dim(2),
        "]",
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim(1),
        ", ",
        b_scales.dim(2),
        "]",
    )

    var a_tma_op = create_tensor_tile[
        Index(1, BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a)

    comptime b_tile_shape = Index(1, BN, BK) if transpose_b else Index(
        1, BK, BN
    )

    var b_tma_op = create_tensor_tile[
        b_tile_shape,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    var a_scales_tma_op = create_tensor_tile[
        Index(1, 1, BM),
        __desc_shape=Index(1, 1, BM),
    ](ctx, a_scales)
    # NOTE: desc shape must be specified otherwise a constraint fails

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[a_scales_type]() * BM

    comptime block_dim = 128

    comptime kernel = _bmm_sm100_blockwise_scaled_fp8_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(a).layout,
        type_of(c).layout,
        type_of(a_scales).layout,
        type_of(b_scales).layout,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c,
        a_scales_tma_op,
        b_scales,
        UInt(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), batch_size),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


def batched_matmul_dynamic_scaled_fp8_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    scales_granularity_mnk: IndexList[3],
    transpose_b: Bool = False,
](
    c_: LayoutTensor[mut=True, c_type, ...],
    a_: LayoutTensor[a_type, ...],
    b_: LayoutTensor[b_type, ...],
    a_scales_: LayoutTensor[a_scales_type, ...],
    b_scales_: LayoutTensor[b_scales_type, ...],
    ctx: DeviceContext,
) raises:
    comptime assert (
        scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == scales_granularity_mnk[2] == 128
    ), "Only support (1,128,128) scale granularity. Extend it for other cases."

    comptime BLOCK_SCALE_K = 128

    # naive implementation requires all tensor have AddressSpace.GENERIC
    var c = c_.address_space_cast[AddressSpace.GENERIC]()
    var a = a_.address_space_cast[AddressSpace.GENERIC]()
    var b = b_.address_space_cast[AddressSpace.GENERIC]()
    var a_scales = a_scales_.address_space_cast[AddressSpace.GENERIC]()
    var b_scales = b_scales_.address_space_cast[AddressSpace.GENERIC]()

    var B = c.dim(0)
    var M = c.dim(1)
    var N = c.dim(2)
    var K = a.dim(2)
    var M_a_scales = a_scales.dim(2)

    # Create 2D layouts by extracting last 2 dims from 3D layouts
    # This preserves the original shape and stride (not assuming row-major)
    comptime c_layout_2d = _2D_layout[c.layout]
    comptime a_layout_2d = _2D_layout[a.layout]
    comptime b_layout_2d = _2D_layout[b.layout]
    comptime a_scales_layout_2d = _2D_layout[a_scales.layout]
    comptime b_scales_layout_2d = _2D_layout[b_scales.layout]

    for batch in range(B):
        # Create 2D LayoutTensor views
        var c_view = LayoutTensor[c_type, c_layout_2d, c.origin](
            c.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[c_layout_2d](
                Index(M, N), Index(c.stride(1), c.stride(2))
            ),
        )
        var a_view = LayoutTensor[a_type, a_layout_2d, a.origin](
            a.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[a_layout_2d](
                Index(M, K), Index(a.stride(1), a.stride(2))
            ),
        )
        var b_view = LayoutTensor[b_type, b_layout_2d, b.origin](
            b.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[b_layout_2d](
                Index(N, K), Index(b.stride(1), b.stride(2))
            ),
        )
        var a_scales_view = LayoutTensor[
            a_scales_type, a_scales_layout_2d, a_scales.origin
        ](
            a_scales.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[a_scales_layout_2d](
                Index(K // BLOCK_SCALE_K, M_a_scales),
                Index(a_scales.stride(1), a_scales.stride(2)),
            ),
        )
        var b_scales_view = LayoutTensor[
            b_scales_type,
            b_scales_layout_2d,
            b_scales.origin,
        ](
            b_scales.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[b_scales_layout_2d](
                Index(N // BLOCK_SCALE_K, K // BLOCK_SCALE_K),
                Index(b_scales.stride(1), b_scales.stride(2)),
            ),
        )

        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
        ](
            c_view,
            a_view,
            b_view,
            a_scales_view,
            b_scales_view,
            ctx,
        )


def batched_matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: LayoutTensor[mut=True, c_type, ...],
    a: LayoutTensor[mut=False, a_type, ...],
    b: LayoutTensor[mut=False, b_type, ...],
    a_scales: LayoutTensor[mut=False, a_scales_type, ...],
    b_scales: LayoutTensor[mut=False, b_scales_type, ...],
    ctx: DeviceContext,
) raises:
    comptime assert (
        _is_sm10x_gpu(ctx.default_device_info)
        or ctx.default_device_info == H100
    ), "Only support SM100 or SM90"
    comptime assert (
        m_scale_granularity == 1
        and n_scale_granularity == 128
        and k_scale_granularity == 128
    ), "Only support (1,128,128) scale granularity"
    comptime assert (
        a_type == b_type == DType.float8_e4m3fn
    ), "input A and B dtype should be float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type == DType.float32
    ), "input A and B scales dtype should be float32"

    comptime assert (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ), "Only support block-wise scale granularity"

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        comptime umma_shape = Index(64, 64, 32)
        comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B

        bmm_sm100_blockwise_scaled_fp8[
            transpose_b=transpose_b,
            umma_shape=umma_shape,
            block_tile_shape=block_tile_shape,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
        ](c, a, b, a_scales, b_scales, ctx)

    else:
        batched_matmul_dynamic_scaled_fp8_naive[
            scales_granularity_mnk=Index(
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
            transpose_b=transpose_b,
        ](c, a, b, a_scales, b_scales, ctx)
