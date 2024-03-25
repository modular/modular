# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_up, div_ceil, max, min, sqrt
from sys._build import is_debug_build
from sys.info import (
    has_avx2,
    has_avx512f,
    has_neon,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_neoverse_n1,
    os_is_macos,
    simdwidthof,
    sizeof,
)

from algorithm import vectorize
from buffer.buffer import (
    DynamicRankBuffer,
    NDBuffer,
    partial_simd_load,
    partial_simd_store,
)
from buffer.list import DimList

from utils.index import Index, StaticIntTuple

alias elementwise_epilogue_type = fn[type: DType, width: Int] (
    StaticIntTuple[2], SIMD[type, width]
) capturing -> None


struct MatmulConfig:
    """Static configuration of tiled matmul algorithms."""

    # Static type info of Operand A.
    var a_type: DType

    # Static shape info of Operand A.
    var a_shape: DimList

    # Static type info of Operand B.
    var b_type: DType

    # Static shape info of Operand B.
    var b_shape: DimList

    # Static type info of Operand C.
    var c_type: DType

    # Static shape info of Operand C.
    var c_shape: DimList

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

    # Indicates if the input matrix A is transposed.
    var transpose_a: Bool

    # Indicates if the input matrix B is transposed.
    var transpose_b: Bool

    # Indicates if the input matrix A is pre-packed.
    var a_packed: Bool

    # Indicates if the input matrix B is pre-packed.
    var b_packed: Bool

    # Enum of the kernel shape, only two shapes currently
    var kernel_type: Bool

    # use AVX_VNNI or AVX512_VNNI
    var use_vnni: Bool

    # use neon_int8_matmul (I8MM)
    var use_i8mm: Bool

    # If true, then perform saturated matmul
    var saturated_vnni: Bool

    fn __init__(
        inout self,
        *,
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
        packed_shape: DimList,
        shape_bias: DimList,
        simd_size: Int,
        a_row_size: Int,
        pack_inner_size: Int,
        pack_data_size: Int,
        prefetch_b_distance_k: Int,
        transpose_a: Bool,
        transpose_b: Bool,
        a_packed: Bool,
        b_packed: Bool,
        kernel_type: Bool,
        use_vnni: Bool,
        use_i8mm: Bool,
        saturated_vnni: Bool,
    ):
        self.a_type = a_type
        self.a_shape = a_shape
        self.b_type = b_type
        self.b_shape = b_shape
        self.c_type = c_type
        self.c_shape = c_shape
        self.packed_shape = packed_shape
        self.shape_bias = shape_bias
        self.simd_size = simd_size
        self.a_row_size = a_row_size
        self.pack_inner_size = pack_inner_size
        self.pack_data_size = pack_data_size
        self.prefetch_b_distance_k = prefetch_b_distance_k
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.a_packed = a_packed
        self.b_packed = b_packed
        self.kernel_type = kernel_type
        self.use_vnni = use_vnni
        self.use_i8mm = use_i8mm
        self.saturated_vnni = saturated_vnni


@value
@register_passable("trivial")
struct MicroKernelShape:
    """Record describing the inner kernel shape."""

    var a_row_size: Int

    var pack_inner_size: Int

    fn __init__(height: Int, width: Int) -> MicroKernelShape:
        return MicroKernelShape {a_row_size: height, pack_inner_size: width}


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
        c_shape: DimList,
        a_shape: DimList,
        b_shape: DimList,
        a_type: DType,
        b_type: DType,
        c_type: DType,
    ](
        c: NDBuffer[c_type, 2, c_shape],
        a: NDBuffer[a_type, 2, a_shape],
        b: NDBuffer[b_type, 2, b_shape],
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
        config: MatmulConfig
    ](
        c: NDBuffer[config.c_type, 2, config.c_shape],
        a: NDBuffer[config.a_type, 2, config.a_shape],
        b: NDBuffer[config.b_type, 2, config.b_shape],
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
            a.dim[0]() if config.transpose_a else a.dim[1](),
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
            index: The int tuple containing the index(m,n,k).

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
@always_inline
fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
    # Factor to adjust for vnni or i8mm
    factor: Int,
](n: Int, k: Int) -> StaticIntTuple[2]:
    """Helper heuristic function to decide on tile size to partition the matmul
    given the cache size and desired data layout.

    Parameters:
        pack_cache_size: Allocated space for packing elements, configuring as a
            function of target cache size desired.
        pack_inner_size: The desired inner dimension of the packed data
            layout.
        factor: Factor to adjust for vnni or i8mm.

    Returns:
        The calculated tile size to partition the matmul as (TileN, TileK).
    """

    var least_tile_n: Int = pack_inner_size

    # Max tile K size based on smallest Tile N.
    var largest_tile_k = align_down(pack_cache_size // least_tile_n, factor)

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.

    var tile_k = min(largest_tile_k, align_up(k, factor))

    # Calculate number of InnerSize to fit in tile_n dimension,
    var max_tile_n_in_inner_size = pack_cache_size // tile_k // pack_inner_size
    var full_data_tile_n_in_inner_size = div_ceil(n, pack_inner_size)
    var tile_n_in_inner_size = min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    var tile_n = tile_n_in_inner_size * pack_inner_size

    return Index(tile_n, tile_k)


fn calculate_tile_n_k[
    # Max number of element to cache.
    pack_cache_size: Int,
    # Inner size of data layout.
    pack_inner_size: Int,
    # Factor to adjust for vnni or i8mm
    factor: Int,
](global_tile_shape: GemmShape) -> StaticIntTuple[2]:
    return calculate_tile_n_k[pack_cache_size, pack_inner_size, factor](
        global_tile_shape.N, global_tile_shape.K
    )


# The number of registers used for the inner kernel is:
#   a_row_size*pack_inner_size + 1*pack_inner_size + 1
fn get_matmul_kernel_shape_x86[kernel_type: Bool]() -> MicroKernelShape:
    @parameter
    if has_avx512f():

        @parameter
        if kernel_type:
            return MicroKernelShape(8, 3)
        else:
            return MicroKernelShape(6, 4)
    else:
        return MicroKernelShape(4, 3)


fn get_matmul_kernel_shape_ARM[
    a_type: DType, b_type: DType, c_type: DType, kernel_type: Bool
]() -> MicroKernelShape:
    @parameter
    if is_neoverse_n1():

        @parameter
        if kernel_type:
            return MicroKernelShape(4, 4)
        else:
            return MicroKernelShape(8, 2)
    else:
        alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

        @parameter
        if use_i8mm:
            return MicroKernelShape(4, 6)
        elif kernel_type:
            return MicroKernelShape(6, 4)
        else:
            return MicroKernelShape(8, 2)


# AVX512 and Neon have 32 registers and AVX has 16.
# The largest kernel for AVX is 4x3 which needs 16 registers and gives the best result.
# For AVX512 a 5x4, 5x5, or 6x4 kernel can be used, 6x4 gives the best result.
# For the Graviton 2 a 8x2 kernel gives the best result in most cases.
# For the Graviton 3 a 6x4 or 4x6 kernel gives the best result.
fn get_matmul_kernel_shape[
    a_type: DType, b_type: DType, c_type: DType, kernel_type: Bool
]() -> MicroKernelShape:
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    @parameter
    if has_neon():
        return get_matmul_kernel_shape_ARM[
            a_type, b_type, c_type, kernel_type
        ]()
    else:
        return get_matmul_kernel_shape_x86[kernel_type]()


fn get_matmul_arch_factor[use_vnni: Bool, use_i8mm: Bool]() -> Int:
    if use_i8mm:
        return 8
    elif use_vnni:
        return 4
    else:
        return 1


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
    a_type: DType,
    b_type: DType,
    c_type: DType,
    simd_size: Int,
    kernel_type: Bool,
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
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()
    var max_row_tasks = div_ceil(m, 2 * kernel_shape.a_row_size)
    var max_col_tasks = div_ceil(n, kernel_shape.pack_inner_size * simd_size)
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
    var num_work_blocks = div_ceil(work, work_block_size)
    var blocks_per_task = num_work_blocks // num_tasks
    var blocks_per_task_extra = num_work_blocks % num_tasks

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
    a_type: DType, b_type: DType, c_type: DType, heuristic: PartitionHeuristic
](
    m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int, kernel_type_m: Int = 0
) -> SubMatmulConfig:
    if get_kernel_type(kernel_type_m, n, k):
        return get_partitioned_matmul[a_type, b_type, c_type, heuristic, True](
            m, n, k, task_id, num_tasks
        )
    else:
        return get_partitioned_matmul[a_type, b_type, c_type, heuristic, False](
            m, n, k, task_id, num_tasks
        )


fn get_partitioned_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    heuristic: PartitionHeuristic,
    kernel_type: Bool,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()

    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    alias a_row_size = kernel_shape.a_row_size
    alias pack_inner_size = kernel_shape.pack_inner_size * simdwidthof[
        DType.float32
    ]()

    @parameter
    if heuristic == PartitionHeuristic.MOJO:

        @parameter
        if use_i8mm:
            # i8mm needs to have even partitions in m.
            # Only the last range is allowed to be odd.
            var partition = get_partitioned_matmul_mojo[
                a_type, b_type, c_type, a_row_size, pack_inner_size, use_i8mm
            ](m // 2, n, k, task_id, num_tasks)

            var t0 = 2 * partition.offset[0]
            var t1 = 2 * partition.shape[0]
            if t0 + t1 == m - 1:
                t1 = m - t0
            partition.offset[0] = t0
            partition.shape[0] = t1
            return partition
        else:
            return get_partitioned_matmul_mojo[
                a_type, b_type, c_type, a_row_size, pack_inner_size
            ](m, n, k, task_id, num_tasks)
    else:
        return get_partitioned_matmul_im2col[a_row_size, pack_inner_size](
            m, n, k, task_id, num_tasks
        )


fn get_partitioned_matmul_mojo[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    micro_kernel_m: Int,
    micro_kernel_n: Int,
    use_i8mm: Bool = False,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    # We can remove version once the new partitioning scheme is accepted.
    alias version = 0
    if version == 0:
        var shape = get_partitioned_matmul_mojo_shape[
            a_type, b_type, c_type, micro_kernel_m, micro_kernel_n, use_i8mm
        ](m, n, k, num_tasks)
        var num_row_tasks = shape[0]
        var num_col_tasks = shape[1]
        var row_task_id = task_id // num_col_tasks
        var col_task_id = task_id % num_col_tasks

        var row_range = partition_work(
            row_task_id, num_row_tasks, m, micro_kernel_m
        )
        var col_range = partition_work(
            col_task_id, num_col_tasks, n, micro_kernel_n
        )
        return SubMatmulConfig(
            Index(row_range[0], col_range[0], 0),
            Index(row_range[1], col_range[1], k),
        )

    else:
        if num_tasks >= 32:
            return get_partitioned_matmul_mojo_v2[
                micro_kernel_m, micro_kernel_n, use_i8mm
            ](m, n, k, task_id, num_tasks)
        else:
            return get_partitioned_matmul_mojo_v1[
                micro_kernel_m, micro_kernel_n, use_i8mm
            ](m, n, k, task_id, num_tasks)


# New partition scheme being developed.
fn get_partitioned_matmul_mojo_shape[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    micro_kernel_m: Int,
    micro_kernel_n: Int,
    use_i8mm: Bool,
](m: Int, n: Int, k: Int, num_tasks: Int) -> StaticIntTuple[2]:
    var num_row_tasks = 1
    var num_col_tasks = 1

    var min_work = m * n

    var num_packs_m = div_ceil(m, micro_kernel_m)
    var num_packs_n = div_ceil(n, micro_kernel_n)
    var max_num_packs_m = num_packs_m
    var max_num_packs_n = num_packs_n
    if (use_i8mm and 2 * m > n) or m > n:
        var half_l2size = get_pack_data_size[b_type]()
        # Limit the partitions in N if the size is smaller than half the L2 cache size.
        var num_packs_n2 = max(k * n // half_l2size, 1)
        if num_packs_m * num_packs_n2 >= num_tasks:
            max_num_packs_n = min(num_packs_n, num_packs_n2)
    else:
        # get the minimum work in n
        var worki = micro_kernel_n * max((num_packs_n // num_tasks), 1)
        # ensure the work in m is not much smaller than in n
        var num_packs_m2 = div_ceil(m, align_down(worki, micro_kernel_m))
        if num_packs_n * num_packs_m2 >= num_tasks:
            max_num_packs_m = min(max_num_packs_m, num_packs_m2)

    max_num_packs_m = min(max_num_packs_m, num_tasks)
    max_num_packs_n = min(max_num_packs_n, num_tasks)
    # Loop over all possible partitions and find the the partition that balances the work best.
    for j in range(max_num_packs_m, 0, -1):
        var workj = micro_kernel_m * div_ceil(num_packs_m, j) if j != 1 else m
        for i in range(min(num_tasks // j, max_num_packs_n), 0, -1):
            var worki = micro_kernel_n * div_ceil(
                num_packs_n, i
            ) if i != 1 else n
            var work = workj * worki
            if work <= min_work:
                min_work = work
                num_row_tasks = j
                num_col_tasks = i

    # heuristic for small m
    if m <= 32 and num_packs_n >= num_tasks:
        num_row_tasks = 1
        num_col_tasks = num_tasks

    return Index(num_row_tasks, num_col_tasks)


# Deprecated partition scheme. Only used for 32 or more threads
fn get_partitioned_matmul_mojo_v2[
    micro_kernel_m: Int, micro_kernel_n: Int, use_i8mm: Bool
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    var h = align_up(m, micro_kernel_m) // micro_kernel_m
    var w = align_up(n, micro_kernel_n) // micro_kernel_n

    var max_threads = min(num_tasks, w * h)
    var threadsm = 1
    var threadsn = 1

    var partition_m: Bool = False

    @parameter
    if use_i8mm:
        if 2 * m > n and h >= max_threads:
            partition_m = True
    else:
        if m > n and h >= max_threads:
            partition_m = True
    if partition_m:
        threadsm = max_threads
    # 2D partitioning does not seem to help when m * k per core is too small.
    # 4096 is an empirical value from looking at several shapes.
    # TODO: find a less arbritary solution for small m*k
    elif w >= max_threads or ((m * k / num_tasks) < 4096):
        threadsn = max_threads
    else:
        # Find the largest threadsm*threadsn value which is <= max_threads.
        # If threadsn1*threadsm1 = threadsn2*threadsm2 and threadsn2>threadsn1
        # then select threadsn2*threadsm2 e.g. 32*2 is prefered to 8*8.
        var jmax = min(max_threads, h) + 1
        var imax = min(max_threads, w) + 1
        var jmin = max_threads // imax
        var imin = max_threads // jmax
        for j in range(jmin, jmax):
            for i in range(imin, imax):
                if i * j <= max_threads:
                    if i * j > threadsn * threadsm:
                        threadsm = j
                        threadsn = i
                    elif i * j == threadsn * threadsm and i > threadsn:
                        threadsm = j
                        threadsn = i

    var row_task_id = task_id // threadsn
    var col_task_id = task_id % threadsn

    var row_range = partition_work(row_task_id, threadsm, m, micro_kernel_m)
    var col_range = partition_work(col_task_id, threadsn, n, micro_kernel_n)
    return SubMatmulConfig(
        Index(row_range[0], col_range[0], 0),
        Index(row_range[1], col_range[1], k),
    )


fn get_partitioned_matmul_mojo_v1[
    micro_kernel_m: Int, micro_kernel_n: Int, use_i8mm: Bool
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    # Based on current performance measurement of DLRM. Row-wise Partition
    # leads to better performance for (m == n and k < m). The reason is not
    # in the shape but how we set cache size (hardcoded for now) and
    # decide tile shape.
    # TODO: generalize if condition once we can configure cache and tiling
    # parameters based on hardwared spec and thread count.

    var partition_m: Bool = False

    @parameter
    if use_i8mm:
        if 2 * m >= n:
            partition_m = True
    else:
        if m > n or (m == n and k <= m):
            partition_m = True

    if partition_m:
        var row_range = partition_work(task_id, num_tasks, m, micro_kernel_m)
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
    var num_packs_n = max(n // micro_kernel_n, 1)
    var num_packs_m = max(m // micro_kernel_m, 1)

    if (
        num_packs_n < 2 * num_col_tasks
        and sqrt(num_packs_m * num_packs_n) > num_col_tasks
    ):
        var num_col_partitions: Int = num_tasks
        var aligned_partition_found: Bool = False
        while num_col_partitions > (num_tasks // num_col_partitions):
            if (
                num_packs_n % num_col_partitions == 0
                and num_tasks % num_col_partitions == 0
            ):
                aligned_partition_found = True
                break
            num_col_partitions -= 1
        # Adjust number of tasks based on partition
        if aligned_partition_found:
            num_col_tasks = num_col_partitions
            num_row_tasks = num_tasks // num_col_tasks

    var row_task_id = task_id // num_col_tasks
    var col_task_id = task_id % num_col_tasks
    var row_range1 = partition_work(
        row_task_id, num_row_tasks, m, micro_kernel_m
    )
    var col_range1 = partition_work(
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
    # Accessing A is more expensive in im2col than accessing B.
    # Time a factor to M to var the heuristic bias on partitioning M.
    # TODO: make this bias factor part of function parameter/argument and
    # unifies interface with matmul partition, e.x. bias=1 for matmul.
    alias bias = 2
    var m_biased = m * bias
    # The ideal partition in theory is to balance the cost of memory access in
    # M and N dimensions using square sub-matrix (after applying the bias).
    var ideal_num_col_tasks = sqrt(div_ceil(n * num_tasks, m_biased))
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
        var eps = div_ceil(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            var num_row_tasks_tmp = num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp > num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    var row_task_id = task_id // num_col_tasks
    var col_task_id = task_id % num_col_tasks
    var row_range = partition_work(
        row_task_id, num_row_tasks, m, micro_kernel_m
    )
    var col_range = partition_work(
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
    if is_debug_build():
        # Only use the large cache size for release build as debug build may
        # contain additional data could cause stack overflow.
        # Restrict it to 4K.
        return 4 * KB // sizeof[type]()

    @parameter
    if os_is_macos():
        # Macos has lower stack limit so lower this allocation too.
        # Restrict it to 64K.
        return 64 * KB // sizeof[type]()

    @parameter
    if has_neon() or has_avx512f():
        # TODO: This should be 1/2 of L2 cache size on Intel. Graviton 2 and
        # Skylake server have a 1 MiB L1 cache AMD Rome has a 512 KiB L2 cache
        # return half the cache size as 4 byte elements
        return 512 * KB // sizeof[type]()

    return 256 * KB // sizeof[type]()


@always_inline
fn get_mm_config[
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
    *,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    a_packed: Bool = False,
    b_packed: Bool = False,
    kernel_type: Bool = False,
    saturated_vnni: Bool = False,
]() -> MatmulConfig:
    """Utility function to extract matmul configuration parameters for exported
    Functions.
        TODO: Add target dependent configuration parameters.
    """
    alias simd_size = simdwidthof[c_type]()

    # number of k iterations to prefetch ahead on the
    #   inner micro kernel loop.
    alias prefetch_b_distance_k = get_matmul_prefetch_b_distance_k()
    alias factor = 4 if use_i8mm_fn[a_type, b_type, c_type]() else simd_size

    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()

    return MatmulConfig(
        a_type=a_type,
        a_shape=a_shape,
        b_type=b_type,
        b_shape=b_shape,
        c_type=c_type,
        c_shape=c_shape,
        packed_shape=DimList.create_unknown[3](),
        shape_bias=DimList.create_unknown[1](),
        simd_size=simd_size,
        a_row_size=kernel_shape.a_row_size,
        pack_inner_size=kernel_shape.pack_inner_size * factor,
        pack_data_size=get_pack_data_size[b_type](),
        prefetch_b_distance_k=prefetch_b_distance_k,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        a_packed=a_packed,
        b_packed=b_packed,
        kernel_type=kernel_type,
        use_vnni=use_vnni_fn[a_type, b_type, c_type](),
        use_i8mm=use_i8mm_fn[a_type, b_type, c_type](),
        saturated_vnni=saturated_vnni,
    )


@always_inline
fn use_vnni_fn[a_type: DType, b_type: DType, c_type: DType]() -> Bool:
    @parameter
    if has_neon_int8_dotprod() and not has_neon_int8_matmul():
        return (
            (a_type == DType.int8 and b_type == DType.int8)
            or (a_type == DType.uint8 and b_type == DType.uint8)
        ) and c_type == DType.int32
    elif has_avx2():
        return (
            a_type == DType.uint8
            and b_type == DType.int8
            and c_type == DType.int32
        )
    else:
        return False


@always_inline
fn use_i8mm_fn[a_type: DType, b_type: DType, c_type: DType]() -> Bool:
    # u8u8, u8s8, s8s8, but not s8u8
    return (
        # Return False for now until i8mm is fully ready.
        has_neon_int8_matmul()
        and (
            (a_type == DType.uint8 and b_type == DType.uint8)
            or (a_type == DType.uint8 and b_type == DType.int8)
            or (a_type == DType.int8 and b_type == DType.int8)
        )
    )


# Determines which kernel shape to use based on the matmul shape MxNxK.
# Currently only allows two shapes.
@always_inline
fn get_kernel_type(m: Int, n: Int, k: Int) -> Bool:
    @parameter
    if has_avx512f():
        return m > 0 and m <= 32
    elif has_neon():

        @parameter
        if is_neoverse_n1():
            return (k % 4096) == 0
        else:
            return m > 32

    else:
        return False


@always_inline
fn get_trace_information(
    name: StringRef,
    shape: GemmShape,
    a_transpose: Bool,
    b_transpose: Bool,
    b_packed: Bool,
) -> String:
    var a_description = String("A=") + shape.M + "x" + shape.K
    var b_description = String("B=") + shape.K + "x" + shape.N
    var c_description = String("C=") + shape.M + "x" + shape.N
    var a_transpose_description = String("a_transpose=") + a_transpose
    var b_transpose_description = String("b_transpose=") + b_transpose
    var b_packed_description = String("b_packed=") + b_packed

    return (
        String(name)
        + ";"
        + a_description
        + ";"
        + b_description
        + ";"
        + c_description
        + ";"
        + a_transpose_description
        + ";"
        + b_transpose_description
        + ";"
        + b_packed_description
    )


fn dispatch_get_kernel_type[
    func: fn[x: Bool] () capturing -> None,
](m: Int, n: Int, k: Int):
    if get_kernel_type(m, n, k):
        func[True]()
    else:
        func[False]()


# TODO(16425): Unify this with the rest of the matmul impl
@always_inline
fn _get_tile_n_k[
    config: MatmulConfig,
    transpose_b: Bool,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    c_type: DType,
    c_shape: DimList,
](b: NDBuffer[b_type, 2, b_shape]) -> StaticIntTuple[2]:
    var tile_n_k: StaticIntTuple[2]
    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

    @parameter
    if not transpose_b:
        tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, factor
        ](b.dim(1), b.dim(0))
    else:
        tile_n_k = calculate_tile_n_k[
            config.pack_data_size, config.pack_inner_size, factor
        ](b.dim(0), b.dim(1))
    return tile_n_k


@always_inline
fn packA_i8mm[
    a_type: DType
](
    t0: Int,
    t1: Int,
    k: Int,
    a_ptr: DTypePointer[a_type],
    a_packed_ptr: DTypePointer[a_type],
):
    @always_inline
    @__copy_capture(k)
    @parameter
    fn packA_helper[nrow: Int](offset: Int):
        var kl = align_down(k, 8)
        var kh = align_up(k, 8)
        var j = t0 + offset
        for l in range(0, k, 8):

            @unroll
            for idx in range(nrow):
                var t0 = a_ptr.load[width=8]((j + idx) * k + l)
                a_packed_ptr.store(kh * j + 2 * l + 8 * idx, t0)

        @unroll
        for idx in range(nrow):
            var t0 = partial_simd_load[8](
                a_ptr.offset((j + idx) * k + kl), 0, k - kl, 0
            )
            partial_simd_store(
                a_packed_ptr.offset(kh * j + 2 * kl + 8 * idx),
                0,
                k - kl,
                t0,
            )

    vectorize[packA_helper, 2](t1 - t0)
