# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from DType import DType
from Index import StaticIntTuple, Index
from Math import div_ceil, max, min, sqrt
from List import DimList
from SIMD import Float32
from TargetInfo import (
    has_avx512f,
    has_neon,
    os_is_macos,
    dtype_simd_width,
    dtype_sizeof,
)
from BuildInfo import is_relwithdebinfo_build, is_debug_build
from Buffer import NDBuffer, DynamicRankBuffer
from SIMD import SIMD

alias elementwise_lambda_fn_sig_type = fn[type: DType, width: Int] (
    StaticIntTuple[2], SIMD[type, width]
) capturing -> None


@register_passable("trivial")
struct MatmulConfig:
    """Static configuration of tiled matmul algorithms."""

    # Static shape info of Operand A.
    var shape_a: DimList

    # Static shape info of Operand B.
    var shape_b: DimList

    # Static shape info of Operand C.
    var shape_c: DimList

    # Static packed shape info of the packed buffer.
    var packed_shape: DimList

    # Static packed shape info of the bias vector.
    var shape_bias: DimList

    # Static info on simd vector size.
    var simd_size: Int

    # Static loop unrolling size on M dimension.
    var a_row_size: Int

    # Static inner dimension of packed data layout.
    var pack_inner_size: Int

    # Static info on number of elements to pack in the packing routine.
    var pack_data_size: Int

    # Prefetch distance for packed b vectors in micro kernels.
    var prefetch_b_distance_k: Int


@register_passable("trivial")
struct MatmulDataType:
    """Record describing the data types of the matrices in a matmul"""

    # The data type of the result (matrix C), and the accumulator.
    var accum_type: DType

    # The data type of the operands (matrix A and B).
    var value_type: DType


@register_passable("trivial")
struct MatmulOperandLayout:
    """Record describing the data layouts of the matmul operands as well as
    intermediate matrices.
    """

    # Indicates if the input matrix A is transposed.
    var transpose_a: Bool

    # Indicates if the input matrix B is transposed.
    var transpose_b: Bool

    # Indicates if the input matrix A is pre-packed.
    var a_packed: Bool

    # Indicates if the input matrix B is pre-packed.
    var b_packed: Bool

    # The inner dimension size for packed A matrix if B is pre-packed.
    var pack_a_inner_size: Int

    # The inner dimension size for packed B matrix if B is pre-packed.
    var pack_b_inner_size: Int


@value
@register_passable("trivial")
struct GemmShape:
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int

    # Construct from dynamic shaped input.
    @staticmethod
    fn get[
        transpose_a: Bool,
        transpose_b: Bool,
        shape_c: DimList,
        shape_a: DimList,
        shape_b: DimList,
        accum_type: DType,
        value_type: DType,
    ](
        c: NDBuffer[2, shape_c, accum_type],
        a: NDBuffer[2, shape_a, value_type],
        b: NDBuffer[2, shape_b, value_type],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        M, N, and K are intentionally calculated using `a` and `c` ONLY. This
        is because `b` may be padded to a multiple of the tile size if it has
        been pre-packed.

        Args:
            c: Buffer with allocated output space.
            a: Buffer containing matrix operand A.
            b: Buffer containing matrix operand B.
        """
        return GemmShape(
            c.dim[0](), c.dim[1](), a.dim[0]() if transpose_a else a.dim[1]()
        )

    @staticmethod
    fn get[
        config: MatmulConfig,
        layout: MatmulOperandLayout,
        data_type: MatmulDataType,
    ](
        c: NDBuffer[2, config.shape_c, data_type.accum_type],
        a: NDBuffer[2, config.shape_a, data_type.value_type],
        b: NDBuffer[2, config.shape_b, data_type.value_type],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        Args:
            c: Buffer with allocated output space.
            a: Buffer containing matrix operand A.
            b: Buffer containing matrix operand B.
        """
        return GemmShape(
            c.dim[0](),
            c.dim[1](),
            a.dim[0]() if layout.transpose_a else a.dim[1](),
        )

    @staticmethod
    fn get(
        c: DynamicRankBuffer,
        a: DynamicRankBuffer,
        b: DynamicRankBuffer,
        transpose_a: Bool,
        transpose_b: Bool,
    ) -> GemmShape:
        return GemmShape(
            c.dim(0),
            c.dim(1),
            a.dim(0) if transpose_a else a.dim(1),
        )

    # TODO: re-enable using StaticIntTuple.
    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        if idx == 0:
            return self.M
        if idx == 1:
            return self.N
        return self.K

    fn __setitem__(inout self, idx: Int, value: Int):
        if idx == 0:
            self.M = value
            return
        if idx == 1:
            self.N = value
            return
        if idx == 2:
            self.K = value
            return

    fn __init__(index: StaticIntTuple[3]) -> GemmShape:
        """Constructor of a gemm shape record from a index tuple.

        Args:
            index (StaticIntTuple): The int tuple containing the index(m,n,k).

        Returns:
            The constructed shape record.
        """
        return GemmShape(
            index[0],
            index[1],
            index[2],
        )

    fn as_index(self) -> StaticIntTuple[3]:
        """Utility to convert the underlying data to an index tuple. So that the
        utilities such as elementwise add can be used.

        Returns:
            The constructed index tuple.
        """
        return Index(self.M, self.N, self.K)

    fn __add__(self, rhs: GemmShape) -> GemmShape:
        """Coordinate-wise addition of two gemm shape records.

        Args:
            rhs: Another gemm shape record to add with.
        """
        return self.as_index() + rhs.as_index()

    fn __sub__(self, rhs: GemmShape) -> GemmShape:
        """Coordinate-wise subtraction of two gemm shape records.

        Args:
            rhs: Another gemm shape record to subtract with.
        """
        return self.as_index() - rhs.as_index()


# Helper heuristic function to decide on tile size
#  Returns (TileN, TileK)
fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
](n: Int, k: Int) -> StaticIntTuple[2]:
    """Helper heuristic function to decide on tile size to partition the matmul
    given the cache size and desired data layout.
        Args:
            pack_cache_size: Allocated space for packing elements, configuring as a
                function of target cache size desired.
            pack_inner_size: The desired inner dimension of the packed data
                layout.
            gemm_shape: The shape of the matmul problem size based on runtime
                input.
        Returns:
            The calculated tile size to partition the matmul as (TileN, TileK)
    """

    let least_tile_n: Int = pack_inner_size

    # Max tile K size based on smallest Tile N.
    let largest_tile_k = pack_cache_size // least_tile_n

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.
    let tile_k = min(largest_tile_k, k)

    # Calculate number of InnerSize to fit in tile_n dimension,
    #  guranteed to be at least 2.
    let max_tile_n_in_inner_size = pack_cache_size // tile_k // pack_inner_size
    let full_data_tile_n_in_inner_size = div_ceil(n, pack_inner_size)
    let tile_n_in_inner_size = min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    let tile_n = tile_n_in_inner_size * pack_inner_size

    return Index(tile_n, tile_k)


fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
](global_tile_shape: GemmShape) -> StaticIntTuple[2]:
    return calculate_tile_n_k[pack_cache_size, pack_inner_size](
        global_tile_shape.N, global_tile_shape.K
    )


# The number of registers used for the inner kernel is:
#   x86:  a_row_size*pack_inner_size + 1*pack_inner_size + 1
#   neon: a_row_size*pack_inner_size + 4*pack_inner_size + 1
# AVX512 has 32 registers and AVX has 16.
# The largest kernel for AVX is 4x3 which needs 16 registers and gives the best result.
# For AVX512 a 5x4, 5x5, or 6x4 kernel can be used, 5x4 gives the best result.
# For the Graviton 2 a 5x3 kernel gives the best result.
fn get_matmul_a_row_size[critical_stride: Bool]() -> Int:
    @parameter
    if has_neon():
        if critical_stride:
            return 4
        else:
            return 8
    elif has_avx512f():
        return 6
    return 4


fn get_matmul_pack_inner_size[critical_stride: Bool]() -> Int:
    @parameter
    if has_neon():
        if critical_stride:
            return 4
        else:
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
    simd_size: Int, critical_stride: Bool
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
    let max_row_tasks = div_ceil(
        m, 2 * get_matmul_a_row_size[critical_stride]()
    )
    let max_col_tasks = div_ceil(
        n, get_matmul_pack_inner_size[critical_stride]() * simd_size
    )
    num_tasks = min(num_tasks, max_row_tasks * max_col_tasks)

    return num_tasks


@value
struct SubMatmulConfig:
    """Static configuration of sub-matrices in parallel matmul."""

    # Starting Indices of sub-matrices.
    var offset: StaticIntTuple[3]

    # Dimension of sub-matrices.
    var shape: StaticIntTuple[3]

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
    heuristic: PartitionHeuristic, critical_stride: Bool
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    # TODO: Add oneDNN/MLAS partition heuristic and use the parameter if below.
    @parameter
    if heuristic == PartitionHeuristic.MOJO:
        return get_partitioned_matmul_mojo[
            get_matmul_a_row_size[critical_stride](),
            get_matmul_pack_inner_size[critical_stride]()
            * dtype_simd_width[DType.float32](),
        ](m, n, k, task_id, num_tasks)
    else:
        return get_partitioned_matmul_im2col[
            get_matmul_a_row_size[critical_stride](),
            get_matmul_pack_inner_size[critical_stride]()
            * dtype_simd_width[DType.float32](),
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
    @noncapturing
    fn int_sqrt_floor(val: Int) -> Int:
        return Int(sqrt(Float32(val)).cast[DType.index]().value)

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


fn get_pack_data_size[type: DType]() -> Int:
    """Utility to compute the number of elements to pack in each tile.
    Returns:
        The number of elements to pack.
    """
    alias KB = 1024

    @parameter
    if is_relwithdebinfo_build() or is_debug_build():
        # Only use the large cache size for release build as debug build may
        # contain additional data could cause stack overflow.
        # Restrict it to 4K.
        return 4 * KB // dtype_sizeof[type]()

    @parameter
    if os_is_macos():
        # Macos has lower stack limit so lower this allocation too.
        # Restrict it to 64K.
        return 64 * KB // dtype_sizeof[type]()

    @parameter
    if has_neon() or has_avx512f():
        # TODO: This should be 1/2 of L2 cache size on Intel. Graviton 2 and
        # Skylake server have a 1 MiB L1 cache AMD Rome has a 512 KiB L2 cache
        # return half the cache size as 4 byte elements
        return 512 * KB // dtype_sizeof[type]()

    return 256 * KB // dtype_sizeof[type]()
