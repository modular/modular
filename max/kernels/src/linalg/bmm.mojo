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
from math import align_up, ceildiv, gcd
from sys import align_of
from sys.info import (
    simd_width_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
)

from algorithm import sync_parallelize, vectorize
from algorithm.functional import _get_start_indices_of_nth_subvolume_uint
from algorithm.reduction import _reduce_generator
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_idx, global_idx
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import is_cpu, is_valid_target, A100
from memory import memset_zero
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from .apple_accelerate import apple_batched_matmul, use_apple_accelerate_lib
from .matmul import _submatmul_sequential_sync
from .matmul_gpu import _matmul_gpu
from .utils import elementwise_epilogue_type as matmul_elementwise_epilogue_type
from .utils import (
    GemmShape,
    get_kernel_config,
    get_kernel_type,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    partition_work,
    use_i8mm_fn,
)
from linalg.matmul_sm100_blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel,
)
from ._multistage_gemm_gpu import multistage_gemm_kernel
from .matmul_amd import gemm_kernel_amd
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout, IntTuple
from buffer import Dim
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
)
from utils.static_tuple import StaticTuple
from layout._ndbuffer_stub import from_ndbuffer_row_major
from sys import size_of
from sys.info import is_nvidia_gpu, is_amd_gpu
from logger import Logger
from gpu.host._nvidia_cuda import TensorMapSwizzle
from layout.tma_async import (
    TMATensorTile,
    create_tma_tile,
)

alias elementwise_epilogue_type = fn[
    c_type: DType,
    width: Int,
    rank: Int,
    *,
    alignment: Int = 1,
] (
    IndexList[rank],
    SIMD[c_type, width],
) capturing -> None


# Similar to _get_start_indices_of_nth_subvolume but returns only the batch
# dimensions for matmul, skipping the last 2 dimsnions.
@always_inline
fn _get_batch_dims[
    rank: Int
](flat_index: Int, shape: IndexList[rank, **_], out res: __type_of(shape)):
    res = {}
    var curr_index = flat_index

    @parameter
    for idx in range(rank - 2):
        # Count from the back, skipping last two dims.
        alias i = rank - idx - 3
        res[i] = curr_index % shape[i]
        curr_index //= shape[i]


# A utility to reshape NDBuffer with rank > 3 to rank-3.
@always_inline
fn _reshape_nd_buffer_with_batch_to_3d(
    buffer: NDBuffer[*_, **_],
) -> NDBuffer[
    buffer.type,
    3,
    buffer.origin,
    DimList(
        Dim(),
        buffer.shape.get[buffer.rank - 2](),
        buffer.shape.get[buffer.rank - 1](),
    ),
    address_space = buffer.address_space,
]:
    alias rank = buffer.rank
    constrained[rank >= 3, "expecting at least rank-3 NDBuffer"]()

    var batch_size = 1

    @parameter
    for i in range(rank - 2):
        batch_size *= buffer.dim[i]()

    var matrix_shape = IndexList[3](
        batch_size, buffer.dim[rank - 2](), buffer.dim[rank - 1]()
    )

    alias static_shape = DimList(
        Dim(),
        buffer.shape.get[buffer.rank - 2](),
        buffer.shape.get[buffer.rank - 1](),
    )

    return NDBuffer[
        buffer.type,
        3,
        buffer.origin,
        static_shape,
        address_space = buffer.address_space,
    ](buffer.data.bitcast[Scalar[buffer.type]](), matrix_shape)


@parameter
fn _reshape_to_3d[layout: Layout]() -> Layout:
    alias rank = len(layout.shape)

    # NOTE: need to cast becasue int tuple returns comptime int
    alias last = Int(layout.shape[rank - 1])
    alias second_last = Int(layout.shape[rank - 2])

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


# A utility to reshape LayoutTensor with rank > 3 to rank-3.
@always_inline
fn _reshape_layout_tensor_with_batch_to_3d[
    c_type: DType,
    c_layout: Layout,
    reshape_layout: Layout = _reshape_to_3d[c_layout](),
](
    tensor: LayoutTensor[c_type, c_layout, *_, **_],
) -> LayoutTensor[
    tensor.dtype,
    reshape_layout,
    tensor.origin,
    address_space = tensor.address_space,
]:
    alias rank = tensor.rank
    constrained[rank >= 3, "expecting at least rank-3 NDBuffer"]()

    var batch_size = 1

    @parameter
    for i in range(rank - 2):
        batch_size *= tensor.dim(i)

    # NOTE: using layout.row_major triggers a debug assert in the indexlist struct, but
    # only for certain shapes

    return LayoutTensor[
        tensor.dtype,
        reshape_layout,
        tensor.origin,
        address_space = tensor.address_space,
    ](
        tensor.ptr,
        RuntimeLayout[reshape_layout](
            {
                batch_size,
                tensor.dim(tensor.rank - 2),
                tensor.dim(tensor.rank - 1),
            },
            {
                tensor.dim(tensor.rank - 2) * tensor.dim(tensor.rank - 1),
                tensor.dim(tensor.rank - 1),
                1,
            },
        ),
    )


# A utility to reshape NDBuffer with rank > 2 to rank-2.
@always_inline
fn _reshape_nd_buffer_with_batch_to_2d(
    buffer: NDBuffer,
) -> NDBuffer[
    buffer.type, 2, buffer.origin, address_space = buffer.address_space
]:
    alias rank = buffer.rank
    constrained[rank >= 2, "expecting at least rank-2 NDBuffer"]()

    var batch_size = 1

    @parameter
    for i in range(rank - 1):
        batch_size *= buffer.dim[i]()

    var matrix_shape = IndexList[2](batch_size, buffer.dim[rank - 1]())

    return NDBuffer[buffer.type, 2, address_space = buffer.address_space](
        buffer.data.bitcast[Scalar[buffer.type]](), matrix_shape
    )


@always_inline
fn _small_batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    elementwise_epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
) raises:
    alias simd_width = simd_width_of[c_type]()

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

            var a_view = NDBuffer[a_type, 1](a_buf.data + batch * K, Index(K))
            var b_view = NDBuffer[b_type, 1](b_buf.data + batch * K, Index(K))

            @always_inline
            @__copy_capture(a_view, b_view)
            @parameter
            fn input_fn[
                dtype: DType, width: Int, rank: Int
            ](idx: IndexList[rank]) -> SIMD[dtype, width]:
                return (
                    a_view.load[width=width](idx[0]).cast[dtype]()
                    * b_view.load[width=width](idx[0]).cast[dtype]()
                ).cast[dtype]()

            @always_inline
            @parameter
            fn output_fn[
                out_type: DType, width: Int, r: Int
            ](i: IndexList[r], value: SIMD[out_type, width]):
                @parameter
                if elementwise_epilogue_fn:
                    alias func = elementwise_epilogue_fn.value()
                    func[out_type, width, rank](indices.canonicalize(), value)
                else:
                    # This will store only once as it is a 1D reduction.
                    # Just use the original [B, B1,...,BN, 0, 0] indices.
                    c_buf.store[width=width](indices, value.cast[c_type]())

            @always_inline
            @parameter
            fn reduce_impl[
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
                    @parameter
                    fn compute_fn[simd_width: Int](n: Int):
                        indices[rank - 1] = n
                        b_buf_index[rank - 1] = n

                        var b_val = b_buf.load[width=simd_width](b_buf_index)

                        c_buf.store[width=simd_width](
                            indices,
                            c_buf.load[width=simd_width](indices)
                            + a_val.cast[c_type]() * b_val.cast[c_type](),
                        )

                    vectorize[compute_fn, simd_width, unroll_factor=2](N)

            @parameter
            if elementwise_epilogue_fn:
                for m in range(M):
                    indices[rank - 2] = m

                    @always_inline
                    @parameter
                    fn apply_epilogue[width: Int](n: Int):
                        indices[rank - 1] = n
                        var val = c_buf.load[width=width](indices)
                        alias func = elementwise_epilogue_fn.value()
                        func[c_type, width, rank](indices, val)

                    vectorize[apply_epilogue, simd_width](N)

    return


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType, //,
    *,
    transpose_a: Bool,
    transpose_b: Bool,
    elementwise_epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: NDBuffer[mut=True, c_type, rank, *_],
    a_buf: NDBuffer[a_type, rank, *_],
    b_buf: NDBuffer[b_type, rank, *_],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    constrained[not transpose_a, "transpose_a not yet supported"]()

    @always_inline
    @parameter
    fn description_fn() -> String:
        # fmt: off
        return String(
            trace_arg("A", a_buf),
            ";", trace_arg("B", b_buf),
            ";", trace_arg("C", c_buf),
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
        @parameter
        if (
            single_thread_blocking_override
            and not transpose_b
            and is_cpu[target]()
        ):
            return _small_batched_matmul[
                rank,
                a_type,
                b_type,
                c_type,
                elementwise_epilogue_fn,
            ](c_buf, a_buf, b_buf)

        batched_matmul[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=elementwise_epilogue_fn,
            saturated_vnni=saturated_vnni,
            target=target,
        ](c_buf, a_buf, b_buf, context=context)


@always_inline
fn _batched_matmul_cpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType, //,
    *,
    transpose_b: Bool,
    elementwise_epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
](
    c_buf: NDBuffer[mut=True, c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
) raises:
    constrained[rank < 5, "max rank for batched matmul is currently 4"]()

    # Batched matmul calls for MacOS >= 13.0.0 and a, b, c of type Float32 are
    # directed to the special Apple-specific implementation.
    @parameter
    if use_apple_accelerate_lib[c_type, a_type, b_type]():
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
    alias min_balance_batch_size = 5
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
    fn task_func(task_id: Int):
        var a_stride_between_batches = a.size() // a.dim[0]()
        var b_stride_between_batches = b.size() // b.dim[0]()
        var c_stride_between_batches = c.size() // c.dim[0]()

        var batch_task_id = task_id // num_tasks_matmul
        var matmul_task_id = task_id % num_tasks_matmul

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
            var c_view = NDBuffer[c_type, 2](
                c.data.offset(batch * c_stride_between_batches),
                IndexList[2](c.dim[1](), c.dim[2]()),
            )
            var a_view = NDBuffer[a_type, 2, MutableAnyOrigin](
                a.data.offset(batch * a_stride_between_batches),
                IndexList[2](a.dim[1](), a.dim[2]()),
            )

            alias config = get_kernel_config[a_type, b_type, c_type]()
            alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
            alias simd_size = config.simd_size
            alias alignment = align_of[SIMD[c_type, simd_size]]()
            var kh = align_up(k, 8)
            var mh = align_up(m, 2)
            var a_packed_ptr = UnsafePointer[Scalar[a_type]]()
            if use_i8mm:
                a_packed_ptr = UnsafePointer[Scalar[a_type]].alloc[
                    alignment=alignment
                ](mh * kh)
            var a_packed = NDBuffer[a_type, 2](a_packed_ptr, DimList(mh, kh))

            if use_i8mm:
                packA_i8mm[a_type](0, m, k, a_view.data, a_packed_ptr)

            var b_view = NDBuffer[b_type, 2](
                b.data.offset(batch * b_stride_between_batches),
                IndexList[2](b.dim[1](), b.dim[2]()),
            )

            var batch_coords = _get_start_indices_of_nth_subvolume_uint[2](
                UInt(batch), c_buf.get_shape()
            )

            @parameter
            fn elementwise_lambda_2d[
                c_type: DType, width: Int, *, alignment: Int = 1
            ](out_coords: IndexList[2], out_val: SIMD[c_type, width]):
                # the caller provided the elementwise epilogue fn over the original
                # buffer rank, not the collapsed buffer rank
                # so un-collapse the batch dims here
                @parameter
                if elementwise_epilogue_fn:
                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    alias func = elementwise_epilogue_fn.value()
                    func[c_type, width, rank](batch_coords, out_val)

            var sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, config.kernel_rows, config.kernel_cols
            ](m, n, k, matmul_task_id, num_tasks_matmul)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            _submatmul_sequential_sync[
                config,
                transpose_b,
                b_packed=False,
                elementwise_lambda_fn = OptionalReg[
                    matmul_elementwise_epilogue_type
                ](elementwise_lambda_2d) if elementwise_epilogue_fn else None,
                saturated_vnni=saturated_vnni,
            ](
                c_view,
                a_packed if use_i8mm else a_view,
                b_view,
                GemmShape(sub_matmul_config.shape),
                GemmShape(sub_matmul_config.offset),
            )
            a_packed_ptr.free()
            _ = batch_coords

    sync_parallelize[task_func](num_tasks)


fn naive_batched_matmul_kernel[
    rank: Int,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
](
    c_tensor: LayoutTensor[c_type, c_layout, MutableAnyOrigin],  # m
    a_tensor: LayoutTensor[a_type, a_layout, MutableAnyOrigin],  # m * k
    b_tensor: LayoutTensor[b_type, b_layout, MutableAnyOrigin],  # 1 * k
    c_buff_nd_shape: IndexList[rank],
) -> None:
    var batch_size: UInt = UInt(c_tensor.dim(0))
    var m: UInt = UInt(c_tensor.dim(1))
    var n: UInt = UInt(c_tensor.dim(2))
    var k: UInt = UInt(a_tensor.dim(2))

    var x = Int(global_idx.x)
    var y = Int(global_idx.y)
    var z = Int(block_idx.z)

    if UInt(z) >= batch_size or UInt(x) >= n or UInt(y) >= m:
        return
    var val = Scalar[accum_type](0)

    alias acc_type = Scalar[accum_type]
    for ki in range(k):
        val += rebind[acc_type](a_tensor[z, y, ki].cast[accum_type]()) * rebind[
            acc_type
        ](b_tensor[z, ki, x].cast[accum_type]())

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        var nd_corrds = _get_start_indices_of_nth_subvolume_uint[2](
            UInt(z), c_buff_nd_shape
        )
        nd_corrds[rank - 1] = x
        nd_corrds[rank - 2] = y
        elementwise_lambda[c_type, 1, rank](nd_corrds, val.cast[c_type]())
    else:
        c_tensor[z, y, x] = val.cast[c_type]()


fn batched_matmul_kernel_gpu[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_tensor: LayoutTensor[c_type, layout_c, MutableAnyOrigin],  # m
    a_tensor: LayoutTensor[a_type, layout_a, MutableAnyOrigin],  # m * k
    b_tensor: LayoutTensor[b_type, layout_b, MutableAnyOrigin],  # 1 * k
    m: Int,
    n: Int,
    k: Int,
):
    var a_ptr = a_tensor.ptr + block_idx.z * (m * k)
    var b_ptr = b_tensor.ptr + block_idx.z * (n * k)
    var c_ptr = c_tensor.ptr + block_idx.z * (m * n)

    alias static_n = b_tensor.shape[1]() if transpose_b else b_tensor.shape[2]()
    alias static_k = b_tensor.shape[2]() if transpose_b else b_tensor.shape[1]()

    alias c_layout = Layout.row_major(UNKNOWN_VALUE, static_n)
    alias a_layout = Layout.row_major(UNKNOWN_VALUE, static_k)
    alias b_layout = Layout.row_major(
        static_n, static_k
    ) if transpose_b else Layout.row_major(static_k, static_n)

    var c = LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        address_space = c_ptr.address_space,
    ](c_ptr, RuntimeLayout[c_layout]({m, n}, {n, 1}))

    var a = LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        address_space = a_ptr.address_space,
    ](a_ptr, RuntimeLayout[a_layout]({m, k}, {k, 1}))

    var b = LayoutTensor[b_type, b_layout](b_ptr)

    @parameter
    fn elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        @parameter
        if elementwise_lambda_fn:
            alias elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_coords = IndexList[3](Int(block_idx.z))
            batch_coords[2] = out_coords[1]
            batch_coords[1] = out_coords[0]
            elementwise_epilogue(batch_coords, val)

    @parameter
    if is_nvidia_gpu():
        multistage_gemm_kernel[
            c_type,
            c.layout,
            a_type,
            a.layout,
            b_type,
            b.layout,
            transpose_b,
            c.layout_int_type,
            a.layout_int_type,
            b.layout_int_type,
            c.linear_idx_type,
            a.linear_idx_type,
            b.linear_idx_type,
            config,
            OptionalReg[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)
    elif is_amd_gpu():
        gemm_kernel_amd[
            c_type,
            c.layout,
            a_type,
            a.layout,
            b_type,
            b.layout,
            transpose_b,
            c.layout_int_type,
            a.layout_int_type,
            b.layout_int_type,
            c.linear_idx_type,
            a.linear_idx_type,
            b.linear_idx_type,
            config,
            OptionalReg[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)


@always_inline
fn get_shape_index_list[
    rank: Int, dtype: DType, layout: Layout
](tensor: LayoutTensor[dtype, layout, *_, **_]) -> IndexList[rank]:
    var index_list = IndexList[rank](0)

    @parameter
    for i in range(rank):
        index_list[i] = tensor.dim(i)
    return index_list


@always_inline
fn _batched_matmul_gpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType, //,
    *,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, c_type, rank, *_, **_],
    a_buf: NDBuffer[a_type, rank, *_, **_],
    b_buf: NDBuffer[b_type, rank, *_, **_],
    ctx: DeviceContext,
) raises:
    var c_tensor = from_ndbuffer_row_major(c_buf)
    var a_tensor = from_ndbuffer_row_major(a_buf)
    var b_tensor = from_ndbuffer_row_major(b_buf)

    var c_tensor_reshaped = _reshape_layout_tensor_with_batch_to_3d(c_tensor)
    var a_tensor_reshaped = _reshape_layout_tensor_with_batch_to_3d(a_tensor)
    var b_tensor_reshaped = _reshape_layout_tensor_with_batch_to_3d(b_tensor)

    var batch_size = c_tensor_reshaped.dim(0)
    var m = c_tensor_reshaped.dim(1)
    var n = c_tensor_reshaped.dim(2)
    var k = a_tensor_reshaped.dim(2)

    alias has_static_NK = b_tensor_reshaped.shape[
        1
    ]() != UNKNOWN_VALUE and b_tensor_reshaped.shape[
        2
    ]() != UNKNOWN_VALUE and a_tensor_reshaped.shape[
        2
    ]() != UNKNOWN_VALUE and c_tensor_reshaped.shape[
        2
    ]() != UNKNOWN_VALUE

    if batch_size == 1:
        with Trace[TraceLevel.OP]("batched_matmul_via_matmul"):
            # If the batch size is 1, then this is just a matmul and we can use the
            # matmul kernel directly.
            @parameter
            if elementwise_epilogue_fn:
                alias elementwise_epilogue = elementwise_epilogue_fn.value()

                @parameter
                @__copy_capture(c_buf)
                fn elementwise_epilogue_fn_wrapper[
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
                ](
                    _reshape_nd_buffer_with_batch_to_2d(c_buf),
                    _reshape_nd_buffer_with_batch_to_2d(a_buf),
                    _reshape_nd_buffer_with_batch_to_2d(b_buf),
                    ctx=ctx,
                )
            else:
                _matmul_gpu[transpose_b=transpose_b](
                    _reshape_nd_buffer_with_batch_to_2d(c_buf),
                    _reshape_nd_buffer_with_batch_to_2d(a_buf),
                    _reshape_nd_buffer_with_batch_to_2d(b_buf),
                    ctx=ctx,
                )

            return

    alias a_k = a_tensor_reshaped.shape[2]()
    alias c_n = c_tensor_reshaped.shape[2]()

    alias multistage_gemm_cond = (
        c_n % 128 == 0 and a_k % 32 == 0 and a_k >= 128
    )

    alias use_A100_kernels = ctx.default_device_info >= A100 and has_nvidia_gpu_accelerator()

    @parameter
    if has_static_NK and use_A100_kernels and multistage_gemm_cond:
        alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

        alias batched_matmul_type = batched_matmul_kernel_gpu[
            c_tensor_reshaped.dtype,
            a_tensor_reshaped.dtype,
            b_tensor_reshaped.dtype,
            c_tensor_reshaped.layout,
            a_tensor_reshaped.layout,
            b_tensor_reshaped.layout,
            transpose_b,
            kernels.ampere_128x128_4,
            elementwise_epilogue_fn,
        ]

        var grid_dim = kernels.ampere_128x128_4.grid_dim(UInt(m), UInt(n))

        ctx.enqueue_function[batched_matmul_type](
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
                kernels.ampere_128x128_4.shared_mem_usage()
            ),
        )
    elif has_static_NK and has_amd_gpu_accelerator() and transpose_b:
        alias block_m = 128
        alias block_n = 128
        alias block_k = 64
        alias config = MatmulConfig[a_type, b_type, c_type, transpose_b](
            block_tile_shape=Index(block_m, block_n, block_k),
            warp_tile_shape=Index(block_m // 2, block_n // 2, block_k),
            num_pipeline_stages=1,
            num_k_partitions=1,
        )
        alias batched_matmul_type = batched_matmul_kernel_gpu[
            c_tensor_reshaped.dtype,
            a_tensor_reshaped.dtype,
            b_tensor_reshaped.dtype,
            c_tensor_reshaped.layout,
            a_tensor_reshaped.layout,
            b_tensor_reshaped.layout,
            transpose_b,
            config,
            elementwise_epilogue_fn,
        ]

        ctx.enqueue_function[batched_matmul_type](
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
    else:
        # TODO: support non-A100 transposed kernels
        constrained[
            not transpose_b,
            "transpose b is not supported on non-A100 capable GPUs",
        ]()

        c_shape = get_shape_index_list[rank, c_type, c_tensor.layout](c_tensor)

        alias BLOCK_DIM = 16
        alias bmm = naive_batched_matmul_kernel[
            rank,
            c_type,
            a_type,
            b_type,
            c_tensor_reshaped.layout,
            a_tensor_reshaped.layout,
            b_tensor_reshaped.layout,
            elementwise_epilogue_fn,
        ]
        ctx.enqueue_function[bmm](
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
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType, //,
    *,
    transpose_b: Bool,
    elementwise_epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: NDBuffer[mut=True, c_type, rank, *_],
    a_buf: NDBuffer[a_type, rank, *_],
    b_buf: NDBuffer[b_type, rank, *_],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    constrained[is_valid_target[target](), "unsupported target"]()

    @parameter
    if is_cpu[target]():
        _batched_matmul_cpu[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=elementwise_epilogue_fn,
            saturated_vnni=saturated_vnni,
        ](c_buf, a_buf, b_buf)
    else:
        constrained[
            saturated_vnni == False,
            "saturated_vnni is not applicable on the gpu",
        ]()
        _batched_matmul_gpu[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=elementwise_epilogue_fn,
        ](c_buf, a_buf, b_buf, context.get_device_context())


@always_inline
fn batched_matmul_shape[
    rank: Int,
    a_type: DType,
    b_type: DType,
    single_thread_blocking_override: Bool,
](
    a_buff: NDBuffer[a_type, rank],
    b_buff: NDBuffer[b_type, rank],
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


alias _2D_layout[layout: Layout] = Layout.row_major(
    layout.shape[1].value(),
    layout.shape[2].value(),
)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
fn _bmm_sm100_blockwise_scaled_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_scales_tile_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c_tensor: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_tile_layout, a_scales_desc_layout
    ],
    b_scales_tensor: LayoutTensor[
        b_scales_type, b_scales_layout, MutableAnyOrigin
    ],
    num_iters: UInt,
):
    alias c_2d_layout: Layout = _2D_layout[c_layout]
    alias b_scales_2d_layout: Layout = _2D_layout[b_scales_layout]

    var M = c_tensor.dim(1)
    var N = c_tensor.dim(2)

    var c_ptr = c_tensor.ptr + (block_idx.z * M * N)
    var b_scales_ptr = b_scales_tensor.ptr + (
        block_idx.z * b_scales_tensor.dim(1) * b_scales_tensor.dim(2)
    )

    var c = LayoutTensor[c_type, c_2d_layout, MutableAnyOrigin](
        c_ptr,
        RuntimeLayout[c_2d_layout].row_major(
            IndexList[2](c_tensor.dim(1), c_tensor.dim(2)),
        ),
    )

    var b_scales = LayoutTensor[
        b_scales_type, b_scales_2d_layout, MutableAnyOrigin
    ](
        b_scales_ptr,
        RuntimeLayout[b_scales_2d_layout].row_major(
            IndexList[2](b_scales_tensor.dim(1), b_scales_tensor.dim(2)),
        ),
    )

    @parameter
    fn elementwise_epilogue_fn_wrapper[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        @parameter
        if elementwise_lambda_fn:
            alias elementwise_epilogue = elementwise_lambda_fn.value()
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
        __type_of(c).layout,
        a_scales_layout,
        __type_of(b_scales).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(a_scales_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(a_scales_tma_op).desc_layout,
        block_tile_shape,
        mma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=num_threads,
        elementwise_lambda_fn = OptionalReg[matmul_elementwise_epilogue_type](
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


fn bmm_sm100_blockwise_scaled_fp8[
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
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, *_, **_],
    a: LayoutTensor[a_type, a_layout, *_, **_],
    b: LayoutTensor[b_type, b_layout, *_, **_],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, *_, **_],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, *_, **_],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    constrained[
        a_type == b_type == DType.float8_e4m3fn,
        "Only support float8_e4m3fn",
    ]()

    constrained[
        b_scales_type == a_scales_type == DType.float32,
        "Only support float32 for a_scales and b_scales",
    ]()

    constrained[
        c.rank == 3,
        "Only support rank 3 tensors",
    ]()

    constrained[
        c.rank == b.rank and c.rank == a.rank,
        "all tensors must have the same rank",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    constrained[BK == 128, "blockwise scaled fp8 only works with BK = 128"]()

    var batch_size = c.dim(0)
    var M = c.dim(1)
    var N = c.dim(2)
    var K = a.dim(2)

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

    var logger = Logger()
    logger.info(
        "Executing Basic 1D2D Blockwise Scaled FP8 GEMM (BLOCK_SCALE_SIZE ="
        " 128)"
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

    var a_tma_op = create_tma_tile[
        a_type,
        3,
        Index(1, BM, BK),
        swizzle_mode=a_swizzle,
        __tile_layout = Layout.row_major(1, BM, BK),
    ](ctx, a)

    alias b_tile_shape = Index(1, BN, BK) if transpose_b else Index(1, BK, BN)

    var b_tma_op = create_tma_tile[
        b_type,
        3,
        b_tile_shape,
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
        __tile_layout = Layout.row_major(b_tile_shape),
    ](ctx, b)

    var a_scales_tma_op = create_tma_tile[
        a_scales_type,
        3,
        Index(1, 1, BM),
        __tile_layout = Layout.row_major(1, 1, BM),
        __desc_layout = Layout(IntTuple(1, 1, BM), IntTuple(1, 1, 1)),
    ](ctx, a_scales)
    # NOTE: desc layout must be specified otherwise a constraint fails

    alias smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[a_scales_type]() * BM

    alias block_dim = 128

    alias kernel = _bmm_sm100_blockwise_scaled_fp8_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        __type_of(a).layout,
        __type_of(c).layout,
        __type_of(a_scales).layout,
        __type_of(b_scales).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(a_scales_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(a_scales_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        a_scales_tma_op,
        b_scales,
        ceildiv(K, BK),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), batch_size),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
