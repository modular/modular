# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import align_up, ceildiv, gcd
from sys import alignof
from sys.info import simdwidthof
from collections.string import StaticString

from algorithm import sync_parallelize, vectorize
from algorithm.functional import (
    _get_start_indices_of_nth_subvolume_uint,
)
from algorithm.reduction import _reduce_generator
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_idx, global_idx
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from memory import UnsafePointer, memset_zero
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type

from .apple_accelerate import apple_batched_matmul, use_apple_accelerate_lib
from .matmul import _submatmul_sequential_sync
from .matmul_gpu import _matmul_gpu
from .utils import elementwise_epilogue_type as matmul_elementwise_epilogue_type
from .utils import (
    get_kernel_type,
    get_kernel_config,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    partition_work,
    use_i8mm_fn,
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
](flat_index: Int, shape: IndexList[rank, **_]) -> __type_of(shape):
    var out = __type_of(shape)()
    var curr_index = flat_index

    @parameter
    for idx in range(rank - 2):
        # Count from the back, skipping last two dims.
        alias i = rank - idx - 3
        out[i] = curr_index % shape[i]
        curr_index //= shape[i]

    return out


# A utility to reshape NDBuffer with rank > 3 to rank-3.
@always_inline
fn _reshape_nd_buffer_with_batch_to_3d(
    buffer: NDBuffer,
) -> NDBuffer[buffer.type, 3, address_space = buffer.address_space]:
    alias rank = buffer.rank
    constrained[rank >= 3, "expecting at least rank-3 NDBuffer"]()

    var batch_size = 1

    @parameter
    for i in range(rank - 2):
        batch_size *= buffer.dim[i]()

    var matrix_shape = IndexList[3](
        batch_size, buffer.dim[rank - 2](), buffer.dim[rank - 1]()
    )

    return NDBuffer[buffer.type, 3, address_space = buffer.address_space](
        buffer.data.bitcast[Scalar[buffer.type]](), matrix_shape
    )


# A utility to reshape NDBuffer with rank > 2 to rank-2.
@always_inline
fn _reshape_nd_buffer_with_batch_to_2d(
    buffer: NDBuffer,
) -> NDBuffer[buffer.type, 2, address_space = buffer.address_space]:
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
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
) raises:
    alias simd_width = simdwidthof[c_type]()

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
                type: DType, width: Int, rank: Int
            ](idx: IndexList[rank]) -> SIMD[type, width]:
                return (
                    a_view.load[width=width](idx[0]).cast[type]()
                    * b_view.load[width=width](idx[0]).cast[type]()
                ).cast[type]()

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
                        func[c_type, width, rank](
                            indices.cast[unsigned=False](), val
                        )

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
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
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
    c_buf: NDBuffer[c_type, rank],
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
        a_type, b_type, c_type, simdwidthof[c_type](), True
    ](m, n, k, num_threads) if get_kernel_type(
        m, n, k
    ) else get_matmul_num_tasks[
        a_type, b_type, c_type, simdwidthof[c_type](), False
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
            var a_view = NDBuffer[a_type, 2](
                a.data.offset(batch * a_stride_between_batches),
                IndexList[2](a.dim[1](), a.dim[2]()),
            )

            alias config = get_kernel_config[a_type, b_type, c_type]()
            alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
            alias simd_size = config.simd_size
            alias alignment = alignof[SIMD[c_type, simd_size]]()
            var kh = align_up(k, 8)
            var mh = align_up(m, 2)
            var a_packed_ptr = UnsafePointer[Scalar[a_type]]()
            if use_i8mm:
                a_packed_ptr = UnsafePointer[
                    Scalar[a_type], alignment=alignment
                ].alloc(mh * kh)
            var a_packed = NDBuffer[a_type, 2](a_packed_ptr, DimList(mh, kh))

            if use_i8mm:
                packA_i8mm[a_type](0, m, k, a_view.data, a_packed_ptr)

            var b_view = NDBuffer[b_type, 2](
                b.data.offset(batch * b_stride_between_batches),
                IndexList[2](b.dim[1](), b.dim[2]()),
            )

            var batch_coords = _get_start_indices_of_nth_subvolume_uint[2](
                batch, c_buf.get_shape()
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
                sub_matmul_config.shape,
                sub_matmul_config.offset,
            )
            a_packed_ptr.free()
            _ = batch_coords

    sync_parallelize[task_func](num_tasks)


fn batched_matmul_kernel[
    rank: Int,
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
](
    c_buff: NDBuffer[c_type, 3, c_shape],
    a_buff: NDBuffer[a_type, 3, a_shape],
    b_buff: NDBuffer[b_type, 3, b_shape],
    c_buff_nd_shape: IndexList[rank],
) -> None:
    var batch_size: UInt = c_buff.dim(0)
    var m: UInt = c_buff.dim(1)
    var n: UInt = c_buff.dim(2)
    var k: UInt = a_buff.dim(2)

    var x = global_idx.x
    var y = global_idx.y
    var z = block_idx.z

    if z >= batch_size or x >= n or y >= m:
        return
    var val = Scalar[accum_type](0)
    for ki in range(k):
        val += (
            a_buff[z, y, ki].cast[accum_type]()
            * b_buff[z, ki, x].cast[accum_type]()
        )

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        var nd_corrds = _get_start_indices_of_nth_subvolume_uint[2](
            z, c_buff_nd_shape
        )
        nd_corrds[rank - 1] = x
        nd_corrds[rank - 2] = y
        elementwise_lambda[c_type, 1, rank](nd_corrds, val.cast[c_type]())
    else:
        c_buff[Index(Int(z), Int(y), Int(x))] = val.cast[c_type]()


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
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
    ctx: DeviceContext,
) raises:
    constrained[not transpose_b, "transpose_b not supported on GPU yet"]()

    var a_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(a_buf)
    var b_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(b_buf)
    var c_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(c_buf)

    var batch_size = a_buf_reshaped.dim[0]()

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
                    type: DType, width: Int, *, alignment: Int = 1
                ](
                    out_coords: IndexList[2], val: SIMD[type, width]
                ) capturing -> None:
                    var batch_coords = IndexList[rank](0)

                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    elementwise_epilogue(batch_coords, val)

                _matmul_gpu[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_epilogue_fn_wrapper,
                ](
                    _reshape_nd_buffer_with_batch_to_2d(c_buf_reshaped),
                    _reshape_nd_buffer_with_batch_to_2d(a_buf_reshaped),
                    _reshape_nd_buffer_with_batch_to_2d(b_buf_reshaped),
                    ctx=ctx,
                )
            else:
                _matmul_gpu[transpose_b=transpose_b](
                    _reshape_nd_buffer_with_batch_to_2d(c_buf_reshaped),
                    _reshape_nd_buffer_with_batch_to_2d(a_buf_reshaped),
                    _reshape_nd_buffer_with_batch_to_2d(b_buf_reshaped),
                    ctx=ctx,
                )

            return

    alias BLOCK_DIM = 16
    alias unkown_shape = DimList.create_unknown[3]()

    var m = a_buf_reshaped.dim[1]()
    var n = b_buf_reshaped.dim[2]()

    alias bmm = batched_matmul_kernel[
        rank,
        c_type,
        unkown_shape,
        a_type,
        unkown_shape,
        b_type,
        unkown_shape,
        elementwise_epilogue_fn,
    ]
    ctx.enqueue_function[bmm](
        c_buf_reshaped,
        a_buf_reshaped,
        b_buf_reshaped,
        c_buf.get_shape(),
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
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: NDBuffer[b_type, rank],
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
