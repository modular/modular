# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param
from DType import DType
from Buffer import NDBuffer
from List import DimList
from LLCL import OutputChainPtr
from Index import StaticIntTuple
from SIMD import SIMD
from Math import max, min, div_ceil, gcd
from MatmulUtils import (
    is_critical_stride,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    PartitionHeuristic,
    partition_work,
)
from Matmul import matmul_sequential_sync
from TargetInfo import dtype_simd_width
from Functional import async_parallelize
from Range import range


@always_inline
fn batched_matmul_parallel_async[
    rank: Int,
    type: DType,
    adj_a: Bool,
    adj_b: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    @parameter
    fn null_elementwise_epilogue[
        type: DType, width: Int
    ](
        out_coords: StaticIntTuple[3],
        out_val: SIMD[type, width],
        c: NDBuffer[3, DimList.create_unknown[3](), type],
    ):
        c.simd_store(out_coords, out_val)

    @closure
    @always_inline
    fn null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[2, DimList.create_unknown[2](), type],
    ):
        pass

    batched_matmul_parallel_async[
        rank,
        type,
        adj_a,
        adj_b,
        False,
        null_elementwise_epilogue,
        False,
    ](c_buf, a_buf, b_buf, null_rowwise_epilogue, out_chain)


@always_inline
fn batched_matmul_parallel_async[
    rank: Int,
    type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_enabled: Bool,
    # fmt: off
    # TODO (#12950): mblack fails to format this signature
    elementwise_epilogue_fn: fn [type: DType, width: Int
        ](StaticIntTuple[3],
          SIMD[type, width],
          NDBuffer[3, DimList.create_unknown[3](), type]
        ) capturing -> None,
    # fmt: on
    rowwise_epilogue_enabled: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), type],
    # fmt: off
    rowwise_epilogue: fn (Int,
                          Int,
                          NDBuffer[2, DimList.create_unknown[2](), type]
                         ) capturing -> None,
    # fmt: on
    out_chain: OutputChainPtr,
):
    assert_param[not adj_a, "batched matmul does not support adj_a yet"]()
    assert_param[rank < 5, "max rank for batched matmul is currently 4"]()
    var batch_size: Int = c_buf.dim[0]()
    if c_buf.get_rank() == 4:
        batch_size *= c_buf.dim(1)

    let c_matrix_shape = StaticIntTuple[3](
        batch_size,
        c_buf.dim(c_buf.get_rank() - 2),
        c_buf.dim(c_buf.get_rank() - 1),
    )
    let a_matrix_shape = StaticIntTuple[3](
        batch_size,
        a_buf.dim(a_buf.get_rank() - 2),
        a_buf.dim(a_buf.get_rank() - 1),
    )
    let b_matrix_shape = StaticIntTuple[3](
        batch_size,
        b_buf.dim(b_buf.get_rank() - 2),
        b_buf.dim(b_buf.get_rank() - 1),
    )

    # Flatten to 3D Tensor.
    let c = NDBuffer[3, DimList.create_unknown[3](), type](
        c_buf.data.bitcast[type](),
        c_matrix_shape,
        type,
    )
    let a = NDBuffer[3, DimList.create_unknown[3](), type](
        a_buf.data.bitcast[type](),
        a_matrix_shape,
        type,
    )
    let b = NDBuffer[3, DimList.create_unknown[3](), type](
        b_buf.data.bitcast[type](),
        b_matrix_shape,
        type,
    )

    let m = c.dim[1]()
    let n = c.dim[2]()
    let k = a.dim[1]() if adj_a else a.dim[2]()
    let num_threads = out_chain.get_runtime().parallelism_level()
    # Prevent parallelizing tiny matrices, e.x. 1024x4x4x4.
    let max_num_tasks_batch = min(
        div_ceil(m * n * k * batch_size, get_min_task_size()), batch_size
    )
    # Prevent parallelizing matmul with too many threads.
    let max_num_tasks_matmul = get_matmul_num_tasks[
        dtype_simd_width[type](), True
    ](m, n, k, num_threads) if is_critical_stride(k) else get_matmul_num_tasks[
        dtype_simd_width[type](), False
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
            let c_view = NDBuffer[2, DimList.create_unknown[2](), type](
                c.data.offset(batch * c_stride_between_batches),
                StaticIntTuple[2](c.dim[1](), c.dim[2]()),
                type,
            )
            let a_view = NDBuffer[2, DimList.create_unknown[2](), type](
                a.data.offset(batch * a_stride_between_batches),
                StaticIntTuple[2](a.dim[1](), a.dim[2]()),
                type,
            )
            let b_view = NDBuffer[2, DimList.create_unknown[2](), type](
                b.data.offset(batch * b_stride_between_batches),
                StaticIntTuple[2](b.dim[1](), b.dim[2]()),
                type,
            )

            @parameter
            fn elementwise_lambda_2d[
                type: DType, width: Int
            ](out_coords: StaticIntTuple[2], out_val: SIMD[type, width]):
                let coords_3d = StaticIntTuple[3](
                    batch, out_coords[0], out_coords[1]
                )
                elementwise_epilogue_fn[type, width](coords_3d, out_val, c)

            @parameter
            @always_inline
            fn rowwise_closure(start_row: Int, num_rows: Int):
                rowwise_epilogue(start_row, num_rows, c_view)

            let sub_matmul_config = get_partitioned_matmul[
                PartitionHeuristic.MOJO
            ](m, n, k, matmul_task_id, num_tasks_matmul)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            matmul_sequential_sync[
                type,
                False,
                adj_b,
                False,  # b_packed
                elementwise_epilogue_enabled,
                elementwise_lambda_2d,
                rowwise_epilogue_enabled,
            ](
                c_view,
                a_view,
                b_view,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
                rowwise_closure,
            )

    async_parallelize[task_func](out_chain, num_tasks)
