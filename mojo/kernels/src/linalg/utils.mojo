# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from DType import DType
from Index import StaticIntTuple, Index
from Math import div_ceil, max, min, sqrt
from SIMD import F32
from TargetInfo import has_avx512f, has_neon, os_is_macos, dtype_simd_width
from BuildInfo import is_relwithdebinfo_build, is_debug_build

# The number of registers used for the inner kernel is:
#   x86:  a_row_size*pack_inner_size + 1*pack_inner_size + 1
#   neon: a_row_size*pack_inner_size + 4*pack_inner_size + 1
# AVX512 has 32 registers and AVX has 16.
# The largest kernel for AVX is 4x3 which needs 16 registers and gives the best result.
# For AVX512 a 5x4, 5x5, or 6x4 kernel can be used, 5x4 gives the best result.
# For the Graviton 2 a 5x3 kernel gives the best result.
fn get_matmul_a_row_size() -> Int:
    @parameter
    if has_neon():
        return 8
    elif has_avx512f():
        return 5
    return 4


fn get_matmul_pack_inner_size() -> Int:
    @parameter
    if has_neon():
        return 2
    elif has_avx512f():
        return 4
    return 3


# prefetching at least on the Graviton 2 performs worse than without.
fn get_matmul_prefetch_b_distance_k() -> Int:
    @parameter
    if has_neon():
        return 0
    return 4


# Min task size. This is copied from MLAS.
# TODO: Replase this magic number with a heuristic based on arch.
fn get_min_task_size() -> Int:
    return 65536


# Unroll factor in packing B
fn get_packB_unroll_factor() -> Int:
    return 8


# ===----------------------------------------------------------------------===#
# Partition Heuristics
# ===----------------------------------------------------------------------===#


@always_inline
fn get_matmul_num_tasks[
    simd_size: Int
](m: Int, n: Int, k: Int, max_num_tasks: Int) -> Int:
    """Compute the number of tasks for parallel matmul.
    The max number of tasks is typically the number of threads/cores."""

    # The min tasks complexity is from MLAS.
    # TODO: We can fine-tune this based on mojo.matmul's scaling.
    var num_tasks = div_ceil(m * n * k, get_min_task_size())
    num_tasks = min(num_tasks, max_num_tasks)

    # Limit num_tasks by row-wise and column-wise partition because we don't
    # support partition in k dim yet. E.x. 32x32x1024 uses 16 threads by min
    # task complexity but we only want it to use <= 4 threads for now since
    # M and N are very small.
    let max_row_tasks = div_ceil(m, 2 * get_matmul_a_row_size())
    let max_col_tasks = div_ceil(n, get_matmul_pack_inner_size() * simd_size)
    num_tasks = min(num_tasks, max_row_tasks * max_col_tasks)

    return num_tasks


struct SubMatmulConfig:
    """Static configuration of sub-matrices in parallel matmul."""

    # Starting Indices of sub-matrices.
    var offset: StaticIntTuple[3]

    # Dimension of sub-matrices.
    var shape: StaticIntTuple[3]

    fn __init__(self&, offset: StaticIntTuple[3], shape: StaticIntTuple[3]):
        self.offset = offset
        self.shape = shape

    fn __copyinit__(self&, existing: Self):
        self.offset = existing.offset
        self.shape = existing.shape

    @always_inline
    fn is_valid(self) -> Bool:
        return self.shape > Index(0, 0, 0)


@register_passable("trivial")
struct PartitionHeuristic:
    var value: Int
    alias MOJO = PartitionHeuristic(0)
    alias Im2col = PartitionHeuristic(1)
    alias ONEDNN = PartitionHeuristic(2)

    @always_inline("nodebug")
    fn __init__(value: Int) -> PartitionHeuristic:
        return PartitionHeuristic {value: value}

    @always_inline("nodebug")
    fn __eq__(self, heuristic: PartitionHeuristic) -> Bool:
        return self.value == heuristic.value


# The work is first grouped into blocks for alignment and load/store efficiency.
# This will partition the work blocks between tasks as even as possible.
@always_inline
fn partition_work(
    task_id: Int, num_tasks: Int, work: Int, work_block_size: Int
) -> StaticIntTuple[2]:
    let num_work_blocks = div_ceil(work, work_block_size)
    let blocks_per_task = num_work_blocks // num_tasks
    let blocks_per_task_extra = num_work_blocks % num_tasks

    var work_per_task = blocks_per_task * work_block_size
    var work_id = (
        work_per_task * task_id + blocks_per_task_extra * work_block_size
    )

    if task_id < blocks_per_task_extra:
        work_per_task = (blocks_per_task + 1) * work_block_size
        work_id = task_id * work_per_task
        return StaticIntTuple[2](work_id, min(work - work_id, work_per_task))

    return StaticIntTuple[2](work_id, min(work - work_id, work_per_task))


fn get_partitioned_matmul[
    heuristic: PartitionHeuristic
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    # TODO: Add oneDNN/MLAS partition heuristic and use the parameter if below.
    @parameter
    if heuristic == PartitionHeuristic.MOJO:
        return get_partitioned_matmul_mojo[
            get_matmul_a_row_size(),
            get_matmul_pack_inner_size() * dtype_simd_width[DType.f32](),
        ](m, n, k, task_id, num_tasks)
    else:
        return get_partitioned_matmul_im2col[
            get_matmul_a_row_size(),
            get_matmul_pack_inner_size() * dtype_simd_width[DType.f32](),
        ](m, n, k, task_id, num_tasks)


fn get_partitioned_matmul_mojo[
    micro_kernel_m: Int,
    micro_kernel_n: Int,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    # Based on current performance measuremnt of DLRM. Row-wise Partition
    # leads to better performance for (m == n and k < m). The reason is not
    # in the shape but how we set cache size (hardcoded for now) and
    # decide tile shape.
    # TODO: generalize if condition once we can configure cache and tiling
    # parameters based on hardwared spec and thread count.
    if m > n or (m == n and k <= m):
        let row_range = partition_work(task_id, num_tasks, m, micro_kernel_m)
        return SubMatmulConfig(
            StaticIntTuple[3](row_range[0], 0, 0),
            StaticIntTuple[3](row_range[1], n, k),
        )

    var num_col_tasks: Int = num_tasks
    var num_row_tasks: Int = 1
    # Try to find a factorization of num_task (aligned partition) where the
    # column partition has multiple pack sizes. This is because it's
    # relatively expensive to handle residual columns when there is only 1-2
    # pack size per partition.
    # We still maintain that there is more column partitions than row
    # partitions since n > m.
    let num_packs = max(n // micro_kernel_n, 1)
    if num_packs < 2 * num_col_tasks:
        var num_col_partitions: Int = num_tasks
        var aligned_partition_found: Bool = False
        while num_col_partitions > (num_tasks // num_col_partitions):
            if (
                num_packs % num_col_partitions == 0
                and num_tasks % num_col_partitions == 0
            ):
                aligned_partition_found = True
                break
            num_col_partitions -= 1
        # Adjust number of tasks based on partition
        if aligned_partition_found:
            num_col_tasks = num_col_partitions
            num_row_tasks = num_tasks // num_col_tasks

    let row_task_id = task_id // num_col_tasks
    let col_task_id = task_id % num_col_tasks
    let row_range1 = partition_work(
        row_task_id, num_row_tasks, m, micro_kernel_m
    )
    let col_range1 = partition_work(
        col_task_id, num_col_tasks, n, micro_kernel_n
    )
    return SubMatmulConfig(
        Index(row_range1[0], col_range1[0], 0),
        Index(row_range1[1], col_range1[1], k),
    )


fn get_partitioned_matmul_im2col[
    micro_kernel_m: Int,
    micro_kernel_n: Int,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    @always_inline
    fn int_sqrt_floor(val: Int) -> Int:
        return Int(sqrt(F32(val)).cast[DType.index]().value)

    # Accessing A is more expensive in im2col than accessing B.
    # Time a factor to M to let the heuristic bias on partitioning M.
    # TODO: make this bias factor part of function parameter/argument and
    # unifies interface with matmul partition, e.x. bias=1 for matmul.
    alias bias = 2
    let m_biased = m * bias
    # The ideal partition in theory is to balance the cost of memory access in
    # M and N dimensions using square sub-matrix (after applying the bias).
    let ideal_num_col_tasks = int_sqrt_floor(div_ceil(n * num_tasks, m_biased))
    var num_row_tasks = num_tasks // ideal_num_col_tasks
    var num_col_tasks = ideal_num_col_tasks

    # Prioritize having at least two packs in N so that A is reused.
    var max_num_col_tasks = min(div_ceil(n, 2 * micro_kernel_n), num_tasks)
    if ideal_num_col_tasks > max_num_col_tasks:
        num_col_tasks = max_num_col_tasks
        num_row_tasks = num_tasks // num_col_tasks
    # In this branch, not all threads get used for ideal_num_col_tasks
    # Check for alternative factorizations use the most threads.
    elif num_tasks % ideal_num_col_tasks != 0:
        # Set 20% deviation.
        # TODO: Make this tuning parameter a function parameter/argument.
        let eps = div_ceil(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            let num_row_tasks_tmp = num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp > num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    let row_task_id = task_id // num_col_tasks
    let col_task_id = task_id % num_col_tasks
    let row_range = partition_work(
        row_task_id, num_row_tasks, m, micro_kernel_m
    )
    let col_range = partition_work(
        col_task_id, num_col_tasks, n, micro_kernel_n
    )
    return SubMatmulConfig(
        Index(row_range[0], col_range[0], 0),
        Index(row_range[1], col_range[1], k),
    )


fn get_pack_data_size() -> Int:
    """Utility to compute the number of elements to pack in each tile.
    Returns:
        The number of elements to pack.
    """

    if is_relwithdebinfo_build() or is_debug_build():
        # Only use the large cache size for release build as debug build may
        #  contain additional data could cause stack overflow.
        return 1024

    if os_is_macos():
        # TODO: macos has lower stack limit so lower this allocation too.
        return 16 * 1024

    # TODO: This should be 1/2 of L2 cache size on Intel.
    # Graviton 2 and Skylake server have a 1 MiB L1 cache
    # AMD Rome has a 512 KiB L2 cache
    # return half the cache size as 4 byte elements
    if has_neon():
        return 128 * 1024
    elif has_avx512f():
        return 128 * 1024
    return 64 * 1024
