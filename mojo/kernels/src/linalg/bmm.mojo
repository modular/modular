# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil, gcd, max, min
from sys.info import simdwidthof

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
)
from memory import memset_zero
from memory.buffer import NDBuffer
from runtime.llcl import OutputChainPtr, OwningOutputChainPtr

from utils.index import Index, StaticIntTuple
from utils.list import DimList


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


@always_inline
fn _small_batched_matmul[
    elementwise_epilogue_enabled: Bool,
    elementwise_epilogue_fn: fn[c_type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    out_chain: OutputChainPtr,
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
                a_buf.data + batch * K, Index(K)
            )
            let b_view = NDBuffer[1, DimList.create_unknown[1](), b_type](
                b_buf.data + batch * K, Index(K)
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
                if elementwise_epilogue_enabled:
                    elementwise_epilogue_fn[out_type, width, rank](
                        indices, value
                    )
                else:
                    # This will store only once as it is a 1D reduction.
                    # Just use the original [B, B1,...,BN, 0, 0] indices.
                    c_buf.simd_store[width](indices, value.cast[c_type]())

            @always_inline
            fn reduce_impl[
                ty: DType, width: Int
            ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
                return v1 + v2

            _reduce_generator[
                c_type,
                1,
                # single_thread_blocking_override,
                True,
                input_fn,
                output_fn,
                reduce_impl,
            ](a_view.dynamic_shape, 0, 0, out_chain)

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
            if elementwise_epilogue_enabled:
                for m in range(M):
                    indices[rank - 2] = m

                    @always_inline
                    @parameter
                    fn apply_epilogue[width: Int](n: Int):
                        indices[rank - 1] = n
                        let val = c_buf.simd_load[width](indices)
                        elementwise_epilogue_fn[c_type, width, rank](
                            indices, val
                        )

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
    elementwise_epilogue_enabled: Bool,
    elementwise_epilogue_fn: fn[c_type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    saturated_vnni: Bool,
    single_thread_blocking_override: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    out_chain: OutputChainPtr,
):
    # TODO: generalize to > rank 3
    @parameter
    if single_thread_blocking_override and not adj_a and not adj_b:
        return _small_batched_matmul[
            elementwise_epilogue_enabled, elementwise_epilogue_fn
        ](c_buf, a_buf, b_buf, out_chain)

    @closure
    @always_inline
    fn null_rowwise_epilogue(
        start_row: Int,
        num_rows: Int,
        c: NDBuffer[2, DimList.create_unknown[2](), c_type],
    ):
        pass

    @parameter
    if single_thread_blocking_override:
        # Any error thrown by this kernel will get swallowed by this chain.
        # (It doesn't presently have any mark_error's)
        let new_chain = OwningOutputChainPtr(out_chain.get_runtime())
        batched_matmul[
            rank,
            a_type,
            b_type,
            c_type,
            adj_a,
            adj_b,
            elementwise_epilogue_enabled,
            elementwise_epilogue_fn,
            rowwise_epilogue_enabled=False,
            saturated_vnni=saturated_vnni,
        ](c_buf, a_buf, b_buf, null_rowwise_epilogue, new_chain.borrow())
        new_chain.wait()
    else:
        batched_matmul[
            rank,
            a_type,
            b_type,
            c_type,
            adj_a,
            adj_b,
            elementwise_epilogue_enabled,
            elementwise_epilogue_fn,
            rowwise_epilogue_enabled=False,
            saturated_vnni=saturated_vnni,
        ](c_buf, a_buf, b_buf, null_rowwise_epilogue, out_chain)


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    out_chain: OutputChainPtr,
):
    @parameter
    fn null_elementwise_epilogue[
        c_type: DType, width: Int, rank: Int
    ](out_coords: StaticIntTuple[rank], out_val: SIMD[c_type, width]):
        pass

    @closure
    @always_inline
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
        False,
        null_elementwise_epilogue,
        False,
    ](c_buf, a_buf, b_buf, null_rowwise_epilogue, out_chain)


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_enabled: Bool,
    elementwise_epilogue_fn: fn[c_type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    rowwise_epilogue_enabled: Bool,
    saturated_vnni: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    rowwise_epilogue: fn (
        Int, Int, NDBuffer[2, DimList.create_unknown[2](), c_type]
    ) capturing -> None,
    out_chain: OutputChainPtr,
):
    constrained[not adj_a, "batched matmul does not support adj_a yet"]()
    constrained[rank < 5, "max rank for batched matmul is currently 4"]()

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
    let c = NDBuffer[3, DimList.create_unknown[3](), c_type](
        c_buf.data.bitcast[c_type](), c_matrix_shape
    )
    let a = NDBuffer[3, DimList.create_unknown[3](), a_type](
        a_buf.data.bitcast[a_type](), a_matrix_shape
    )
    let b = NDBuffer[3, DimList.create_unknown[3](), b_type](
        b_buf.data.bitcast[b_type](), b_matrix_shape
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
        simdwidthof[c_type](), True
    ](m, n, k, num_threads) if is_critical_stride(k) else get_matmul_num_tasks[
        simdwidthof[c_type](), False
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
                var coords = _get_start_indices_of_nth_subvolume[rank, 2](
                    batch, c_buf.get_shape()
                )
                coords[rank - 1] = out_coords[1]
                coords[rank - 2] = out_coords[0]

                elementwise_epilogue_fn[c_type, width, rank](coords, out_val)

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

            _submatmul_sequential_sync[
                a_type,
                DimList.create_unknown[2](),
                b_type,
                DimList.create_unknown[2](),
                c_type,
                DimList.create_unknown[2](),
                False,
                adj_b,
                False,  # b_packed
                elementwise_epilogue_enabled,
                elementwise_lambda_2d,
                rowwise_epilogue_enabled,
                saturated_vnni,
            ](
                c_view,
                a_view,
                b_view,
                sub_matmul_config.shape,
                sub_matmul_config.offset,
                rowwise_closure,
            )

    sync_parallelize[task_func](out_chain, num_tasks)


@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    adj_a: Bool,
    adj_b: Bool,
    elementwise_epilogue_enabled: Bool,
    elementwise_epilogue_fn: fn[c_type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[c_type, width]
    ) capturing -> None,
    rowwise_epilogue_enabled: Bool,
](
    c_buf: NDBuffer[rank, DimList.create_unknown[rank](), c_type],
    a_buf: NDBuffer[rank, DimList.create_unknown[rank](), a_type],
    b_buf: NDBuffer[rank, DimList.create_unknown[rank](), b_type],
    rowwise_epilogue: fn (
        Int, Int, NDBuffer[2, DimList.create_unknown[2](), c_type]
    ) capturing -> None,
    out_chain: OutputChainPtr,
):
    batched_matmul[
        rank,
        a_type,
        b_type,
        c_type,
        adj_a,
        adj_b,
        elementwise_epilogue_enabled,
        elementwise_epilogue_fn,
        rowwise_epilogue_enabled,
        saturated_vnni=False,
    ](c_buf, a_buf, b_buf, rowwise_epilogue, out_chain)


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
