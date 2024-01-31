# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil, gcd, max, min
from sys.info import simdwidthof


from gpu import ThreadIdx, BlockIdx, BlockDim
from gpu.host import Function, Stream

from algorithm import sync_parallelize, vectorize_unroll
from algorithm.functional import _get_start_indices_of_nth_subvolume
from algorithm.reduction import _reduce_generator
from Matmul import _submatmul_sequential_sync
from MatmulUtils import (
    PartitionHeuristic,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    is_critical_stride,
    partition_work,
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from memory import memset_zero
from memory.buffer import NDBuffer
from runtime.llcl import Runtime

from utils.index import StaticIntTuple
from utils.list import DimList
from utils._optional import Optional

alias elementwise_epilogue_type = fn[c_type: DType, width: Int, rank: Int] (
    StaticIntTuple[rank], SIMD[c_type, width]
) capturing -> None


# Similar to _get_start_indices_of_nth_subvolume but returns only the batch
# dimensions for matmul, skipping the last 2 dimsnions.
@always_inline
fn _get_batch_dims[
    rank: Int
](flat_index: Int, shape: StaticIntTuple[rank]) -> StaticIntTuple[rank]:
    var out = StaticIntTuple[rank]()
    var curr_index = flat_index

    @always_inline
    @parameter
    fn compute_shape[idx: Int]():
        # Count from the back, skipping last two dims.
        alias i = rank - idx - 3
        out[i] = curr_index % shape[i]
        curr_index //= shape[i]

    unroll[rank - 2, compute_shape]()
    return out


# A utility to reshape NDBuffer with rank > 3 to rank-3.
@always_inline
fn _reshape_nd_buffer_with_batch_to_3d[
    rank: Int, dtype: DType
](buffer: NDBuffer[rank, DimList.create_unknown[rank](), dtype]) -> NDBuffer[
    3, DimList.create_unknown[3](), dtype
]:
    constrained[rank >= 3, "expecting at least rank-3 NDBuffer"]()

    var batch_size = 1
    for i in range(0, rank - 2):
        batch_size *= buffer.dim(i)

    let matrix_shape = StaticIntTuple[3](
        batch_size,
        buffer.dim(buffer.get_rank() - 2),
        buffer.dim(buffer.get_rank() - 1),
    )

    return NDBuffer[3, DimList.create_unknown[3](), dtype](
        buffer.data.bitcast[dtype](), matrix_shape
    )


@always_inline
fn _small_batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
):
    alias simd_width = simdwidthof[c_type]()

    # Get the flattened batch.
    var batch_shape = c_buf.dynamic_shape
    batch_shape[rank - 2] = 1
    batch_shape[rank - 1] = 1
    let B = batch_shape.flattened_length()

    let M = a_buf.dim[rank - 2]()
    let N = b_buf.dim[rank - 1]()
    let K = a_buf.dim[rank - 1]()

    if M == 1 and N == 1:
        for batch in range(B):
            # Get the indices as (B1, B2, ..., BN, 0, 0) where B is
            # each trailing batch dimension.
            var indices = _get_batch_dims[rank](batch, c_buf.dynamic_shape)

            let a_view = NDBuffer[1, DimList.create_unknown[1](), a_type](
                a_buf.data + batch * K, (K)
            )
            let b_view = NDBuffer[1, DimList.create_unknown[1](), b_type](
                b_buf.data + batch * K, (K)
            )

            @always_inline
            @parameter
            fn input_fn[
                type: DType, width: Int, rank: Int
            ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
                return (
                    a_view.simd_load[width](idx[0]).cast[type]()
                    * b_view.simd_load[width](idx[0]).cast[type]()
                ).cast[type]()

            @always_inline
            @parameter
            fn output_fn[
                out_type: DType, width: Int, r: Int
            ](i: StaticIntTuple[r], value: SIMD[out_type, width]):
                @parameter
                if elementwise_epilogue_fn:
                    alias func = elementwise_epilogue_fn.value()
                    func[out_type, width, rank](indices, value)
                else:
                    # This will store only once as it is a 1D reduction.
                    # Just use the original [B, B1,...,BN, 0, 0] indices.
                    c_buf.simd_store[width](indices, value.cast[c_type]())

            @always_inline
            @parameter
            fn reduce_impl[
                ty: DType, width: Int
            ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
                return v1 + v2

            try:
                _reduce_generator[
                    input_fn,
                    output_fn,
                    reduce_impl,
                    single_thread_blocking_override=True,
                ](
                    a_view.dynamic_shape,
                    init=Scalar[c_type](0),
                    reduce_dim=0,
                )
            except e:
                trap(e)

    else:
        for batch in range(B):
            # Get the indices as (B1, B2, ..., BN, 0, 0) where B is
            # each trailing batch dimension.
            var indices = _get_batch_dims[rank](batch, c_buf.dynamic_shape)
            var b_buf_index = indices

            memset_zero(c_buf.data + batch * M * N, M * N)
            for m in range(M):
                indices[rank - 2] = m

                for k in range(K):
                    indices[rank - 1] = k
                    b_buf_index[rank - 2] = k

                    let a_val = a_buf[indices]

                    @always_inline
                    @parameter
                    fn compute_fn[simd_width: Int](n: Int):
                        indices[rank - 1] = n
                        b_buf_index[rank - 1] = n

                        let b_val = b_buf.simd_load[simd_width](b_buf_index)

                        c_buf.simd_store[simd_width](
                            indices,
                            c_buf.simd_load[simd_width](indices)
                            + a_val.cast[c_type]() * b_val.cast[c_type](),
                        )

                    alias unroll_factor = 2
                    vectorize_unroll[simd_width, unroll_factor, compute_fn](N)

            @parameter
            if elementwise_epilogue_fn:
                for m in range(M):
                    indices[rank - 2] = m

                    @always_inline
                    @parameter
                    fn apply_epilogue[width: Int](n: Int):
                        indices[rank - 1] = n
                        let val = c_buf.simd_load[width](indices)
                        alias func = elementwise_epilogue_fn.value()
                        func[c_type, width, rank](indices, val)

                    vectorize_unroll[simd_width, 1, apply_epilogue](N)

    return


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
):
    # TODO: generalize to > rank 3
    @parameter
    if (
        single_thread_blocking_override
        and not adj_a
        and not adj_b
        and target == "cpu"
    ):
        return _small_batched_matmul[
            rank,
            a_type,
            b_type,
            c_type,
            elementwise_epilogue_fn,
        ](c_buf, a_buf, b_buf)

    fn null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[2, DimList.create_unknown[2](), c_type],
    ):
        pass

    batched_matmul[
        rank,
        a_type,
        b_type,
        c_type,
        adj_a,
        adj_b,
        elementwise_epilogue_fn,
        rowwise_epilogue_enabled=False,
        saturated_vnni=saturated_vnni,
        target=target,
    ](c_buf, a_buf, b_buf, null_rowwise_epilogue)


@always_inline
fn _batched_matmul_cpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    rowwise_epilogue_enabled: Bool = False,
    saturated_vnni: Bool = False,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    rowwise_epilogue: fn (
        Int, Int, NDBuffer[2, DimList.create_unknown[2](), c_type]
    ) escaping -> None,
):
    constrained[not adj_a, "batched matmul does not support adj_a yet"]()
    constrained[rank < 5, "max rank for batched matmul is currently 4"]()

    # Flatten to 3D Tensor.
    let c = _reshape_nd_buffer_with_batch_to_3d(c_buf)
    let a = _reshape_nd_buffer_with_batch_to_3d(a_buf)
    let b = _reshape_nd_buffer_with_batch_to_3d(b_buf)
    var batch_size: Int = c.dim[0]()

    let m = c.dim[1]()
    let n = c.dim[2]()
    let k = a.dim[1]() if adj_a else a.dim[2]()
    let num_threads = Runtime().parallelism_level()
    # Prevent parallelizing tiny matrices, e.x. 1024x4x4x4.
    let max_num_tasks_batch = min(
        div_ceil(m * n * k * batch_size, get_min_task_size()), batch_size
    )
    # Prevent parallelizing matmul with too many threads.
    let max_num_tasks_matmul = get_matmul_num_tasks[
        a_type, b_type, c_type, simdwidthof[c_type](), True
    ](m, n, k, num_threads) if is_critical_stride(k) else get_matmul_num_tasks[
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
    let batch_size_per_task = batch_size // num_tasks_batch_tmp
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

    let num_tasks_batch = num_tasks_batch_tmp
    let num_tasks_matmul = num_tasks_matmul_tmp
    let num_tasks = num_tasks_batch * num_tasks_matmul

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        let a_stride_between_batches = a.size() // a.dim[0]()
        let b_stride_between_batches = b.size() // b.dim[0]()
        let c_stride_between_batches = c.size() // c.dim[0]()

        let batch_task_id = task_id // num_tasks_matmul
        let matmul_task_id = task_id % num_tasks_matmul

        let num_batches = c.dim[0]()
        # Set the granularity to 1 to divide the batches among tasks
        # as even as possible.
        let batch_range = partition_work(
            batch_task_id, num_tasks_batch, num_batches, 1
        )
        let batch_start = batch_range[0]
        let batches_per_task = batch_range[1]

        # Partition the matmul

        for batch in range(batch_start, batch_start + batches_per_task):
            # Get a 2D view of the 3D Tensor.
            let c_view = NDBuffer[2, DimList.create_unknown[2](), c_type](
                c.data.offset(batch * c_stride_between_batches),
                StaticIntTuple[2](c.dim[1](), c.dim[2]()),
            )
            let a_view = NDBuffer[2, DimList.create_unknown[2](), a_type](
                a.data.offset(batch * a_stride_between_batches),
                StaticIntTuple[2](a.dim[1](), a.dim[2]()),
            )
            let b_view = NDBuffer[2, DimList.create_unknown[2](), b_type](
                b.data.offset(batch * b_stride_between_batches),
                StaticIntTuple[2](b.dim[1](), b.dim[2]()),
            )

            @parameter
            fn elementwise_lambda_2d[
                c_type: DType, width: Int
            ](out_coords: StaticIntTuple[2], out_val: SIMD[c_type, width]):
                # the caller provided the elementwise epilogue fn over the original
                # buffer rank, not the collapsed buffer rank
                # so un-collapse the batch dims here
                @parameter
                if elementwise_epilogue_fn:
                    var coords = _get_start_indices_of_nth_subvolume[rank, 2](
                        batch, c_buf.get_shape()
                    )
                    coords[rank - 1] = out_coords[1]
                    coords[rank - 2] = out_coords[0]

                    alias func = elementwise_epilogue_fn.value()
                    func[c_type, width, rank](coords, out_val)

            fn rowwise_closure(start_row: Int, num_rows: Int):
                rowwise_epilogue(start_row, num_rows, c_view)

            let sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, PartitionHeuristic.MOJO
            ](m, n, k, matmul_task_id, num_tasks_matmul)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            _submatmul_sequential_sync[
                a_type,
                DimList.create_unknown[2](),
                b_type,
                DimList.create_unknown[2](),
                c_type,
                DimList.create_unknown[2](),
                transpose_a=False,
                transpose_b=adj_b,
                b_packed=False,
                elementwise_lambda_fn = Optional[
                    matmul_elementwise_epilogue_type
                ](elementwise_lambda_2d) if elementwise_epilogue_fn else None,
                rowwise_epilogue_enabled=rowwise_epilogue_enabled,
                saturated_vnni=saturated_vnni,
            ](
                c_view,
                a_view,
                b_view,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
                rowwise_closure,
            )

    sync_parallelize[task_func](num_tasks)


fn batched_matmul_kernel[
    rank: Int,
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buff: NDBuffer[3, c_shape, c_type],
    a_buff: NDBuffer[3, a_shape, a_type],
    b_buff: NDBuffer[3, b_shape, b_type],
    c_buff_nd_shape: StaticIntTuple[rank],
) -> None:
    let batch_size = c_buff.dim(0)
    let m = c_buff.dim(1)
    let n = c_buff.dim(2)
    let k = a_buff.dim(2)

    let x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    let y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()
    let z = BlockIdx.z()

    if z >= batch_size or x >= n or y >= m:
        return
    var val = Scalar[c_type](0.0)
    for ki in range(k):
        val += a_buff[z, y, ki].cast[c_type]() * b_buff[z, ki, x].cast[c_type]()

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        var nd_corrds = _get_start_indices_of_nth_subvolume[rank, 2](
            z, c_buff_nd_shape
        )
        nd_corrds[rank - 1] = x
        nd_corrds[rank - 2] = y
        elementwise_lambda[c_type, 1, rank](nd_corrds, val)
    else:
        c_buff[(z, y, x)] = val


@always_inline
fn _batched_matmul_gpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    rowwise_epilogue_enabled: Bool = False,
    saturated_vnni: Bool = False,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    rowwise_epilogue: fn (
        Int, Int, NDBuffer[2, DimList.create_unknown[2](), c_type]
    ) escaping -> None,
):
    constrained[
        not rowwise_epilogue_enabled, "rowwise epilogue fusion isn't supported"
    ]()

    let a_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(a_buf)
    let b_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(b_buf)
    let c_buf_reshaped = _reshape_nd_buffer_with_batch_to_3d(c_buf)

    alias BLOCK_DIM = 16
    alias unkown_shape = DimList.create_unknown[3]()

    let batch_size = a_buf_reshaped.dim(0)
    let m = a_buf_reshaped.dim(1)
    let k = a_buf_reshaped.dim(2)
    let n = b_buf_reshaped.dim(2)

    try:
        let stream = Stream.get_current_stream()
        let gpu_func = Function[
            fn (
                NDBuffer[3, unkown_shape, c_type],
                NDBuffer[3, unkown_shape, a_type],
                NDBuffer[3, unkown_shape, b_type],
                StaticIntTuple[rank],
            ) capturing -> None, batched_matmul_kernel[
                rank,
                c_type,
                unkown_shape,
                a_type,
                unkown_shape,
                b_type,
                unkown_shape,
                elementwise_epilogue_fn,
            ]
        ]()
        gpu_func(
            stream,
            (div_ceil(n, BLOCK_DIM), div_ceil(m, BLOCK_DIM), batch_size),
            (BLOCK_DIM, BLOCK_DIM, 1),
            c_buf_reshaped,
            a_buf_reshaped,
            b_buf_reshaped,
            c_buf.dynamic_shape,
        )
    except e:
        trap(e)


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    rowwise_epilogue_enabled: Bool = False,
    saturated_vnni: Bool = False,
    target: StringLiteral = "cpu",
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    rowwise_epilogue: fn (
        Int, Int, NDBuffer[2, DimList.create_unknown[2](), c_type]
    ) escaping -> None,
):
    constrained[target == "cpu" or target == "cuda", "unsupported target"]()
    alias func = _batched_matmul_cpu if target == "cpu" else _batched_matmul_gpu
    func[
        rank,
        a_type,
        b_type,
        c_type,
        adj_a,
        adj_b,
        elementwise_epilogue_fn,
        rowwise_epilogue_enabled,
        saturated_vnni,
    ](c_buf, a_buf, b_buf, rowwise_epilogue)


@always_inline
fn batched_matmul_shape[
    rank: Int,
    a_type: DType,
    b_type: DType,
    single_thread_blocking_override: Bool,
](
    a_buff: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buff: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
) raises -> StaticIntTuple[rank]:
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
        raise Error("batched_matmul requires rank >= 2")

    if a_buff.dim(rank - 1) != b_buff.dim(rank - 2):
        raise Error("batched_matmul inner dimension must match")

    # Check batch dimensions
    var foundMismatch = False

    # TODO bring this back once `SymbolicizeFallbackShapeFunctions` can handle
    # multipl asserts.
    # @unroll
    # for i in range(rank - 2):
    #    if a_buff.dim(i) != b_buff.dim(i):
    #        foundMismatch = True

    if foundMismatch:
        raise Error("batched_matmul batch dimensions must match")

    var output_shape = a_buff.get_shape()
    output_shape[rank - 1] = b_buff.dim(rank - 1)

    return output_shape


@always_inline
fn get_trace_information[
    rank: Int
](
    name: StringRef,
    a_matrix_shape: StaticIntTuple[rank],
    b_matrix_shape: StaticIntTuple[rank],
    c_matrix_shape: StaticIntTuple[rank],
    adj_a: Bool,
    adj_b: Bool,
) -> String:
    let shape_a: String
    let shape_b: String
    let shape_c: String
    shape_a = String("x").join(a_matrix_shape)
    shape_b = String("x").join(b_matrix_shape)
    shape_c = String("x").join(c_matrix_shape)

    let a_description = String("A=") + shape_a
    let b_description = String("B=") + shape_b
    let c_description = String("C=") + shape_c
    let adj_a_description = String("adj_a=") + adj_a
    let adj_b_description = String("adj_b=") + adj_b

    return String(";").join(
        String(name),
        a_description,
        b_description,
        c_description + ";" + adj_a_description + ";" + adj_b_description,
    )
