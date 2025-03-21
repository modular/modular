# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_up, ceildiv
from sys import alignof
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
from buffer.buffer import NDBuffer, partial_simd_load, partial_simd_store
from buffer.dimlist import DimList
from layout.layout import *
from layout.layout_tensor import LayoutTensor
from memory import UnsafePointer

from utils.index import Index, IndexList

alias elementwise_epilogue_type = fn[
    type: DType, width: Int, *, alignment: Int = 1
] (IndexList[2], SIMD[type, width]) capturing -> None


struct KernelConfig:
    """Static configuration of the matmul inner kernel."""

    # Static number of rows of the micro kernel.
    var kernel_rows: Int

    # Static number of columns of the micro kernel.
    var kernel_cols: Int

    # Static info on simd vector size.
    var simd_size: Int

    # Static packed shape info of the packed buffer.
    var packed_shape: DimList

    fn __init__(
        mut self,
        *,
        kernel_rows: Int,
        kernel_cols: Int,
        simd_size: Int,
        packed_shape: DimList,
    ):
        self.kernel_rows = kernel_rows
        self.kernel_cols = kernel_cols
        self.simd_size = simd_size
        self.packed_shape = packed_shape


@value
@register_passable("trivial")
struct MicroKernelShape:
    """Record describing the inner kernel shape."""

    var simd_rows: Int

    var simd_cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.simd_rows = rows
        self.simd_cols = cols


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
        transpose_b: Bool,
    ](
        c: NDBuffer[rank=2, *_, **_],
        a: NDBuffer[rank=2, *_, **_],
        b: NDBuffer[rank=2, *_, **_],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        M, N, and K are intentionally calculated using `a` and `c` ONLY. This
        is because `b` may be padded to a multiple of the tile size if it has
        been pre-packed.

        Args:
            c: NDBuffer with allocated output space.
            a: NDBuffer containing matrix operand A.
            b: NDBuffer containing matrix operand B.
        """
        return GemmShape(c.dim[0](), c.dim[1](), a.dim[1]())

    # TODO: re-enable using IndexList.
    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        if idx == 0:
            return self.M
        if idx == 1:
            return self.N
        return self.K

    fn __setitem__(mut self, idx: Int, value: Int):
        if idx == 0:
            self.M = value
            return
        if idx == 1:
            self.N = value
            return
        if idx == 2:
            self.K = value
            return

    @implicit
    fn __init__(out self, index: IndexList[3]):
        """Constructor of a gemm shape record from a index tuple.

        Args:
            index: The int tuple containing the index(m,n,k).
        """
        self.M = index[0]
        self.N = index[1]
        self.K = index[2]

    fn as_index(self) -> IndexList[3]:
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
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
](n: Int, k: Int) -> IndexList[2]:
    """Helper heuristic function to decide on tile size to partition the matmul
    given the cache size and desired data layout.

    Parameters:
        a_type: The type of the A tensor.
        b_type: The type of the B tensor.
        c_type: The type of the C tensor.
        kernel_cols: The umber of columns of the micro kernel.

    Returns:
        The calculated tile size to partition the matmul as (TileN, TileK).
    """

    alias pack_cache_size = get_pack_data_size[b_type]()
    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

    var least_tile_n: Int = kernel_cols

    # Max tile K size based on smallest Tile N.
    var largest_tile_k = align_down(pack_cache_size // least_tile_n, factor)

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.

    var tile_k = min(largest_tile_k, align_up(k, factor))

    # Calculate number of InnerSize to fit in tile_n dimension,
    var max_tile_n_in_inner_size = pack_cache_size // tile_k // kernel_cols
    var full_data_tile_n_in_inner_size = ceildiv(n, kernel_cols)
    var tile_n_in_inner_size = min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    var tile_n = tile_n_in_inner_size * kernel_cols

    return Index(tile_n, tile_k)


fn calculate_tile_n_k[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
](global_tile_shape: GemmShape) -> IndexList[2]:
    return calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
        global_tile_shape.N, global_tile_shape.K
    )


@always_inline
fn _get_tile_n_k[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
    transpose_b: Bool,
](b: NDBuffer[_, 2, _, _]) -> IndexList[2]:
    var tile_n_k: IndexList[2]

    @parameter
    if not transpose_b:
        tile_n_k = calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
            b.dim(1), b.dim(0)
        )
    else:
        tile_n_k = calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
            b.dim(0), b.dim(1)
        )
    return tile_n_k


# The number of registers used for the inner kernel is:
#   kernel_rows*kernel_cols + 1*kernel_cols + 1
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


# ===-----------------------------------------------------------------------===#
# Partition Heuristics
# ===-----------------------------------------------------------------------===#


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
    var num_tasks = ceildiv(m * n * k, get_min_task_size())
    num_tasks = min(num_tasks, max_num_tasks)

    # Limit num_tasks by row-wise and column-wise partition because we don't
    # support partition in k dim yet. E.x. 32x32x1024 uses 16 threads by min
    # task complexity but we only want it to use <= 4 threads for now since
    # M and N are very small.
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()
    var max_row_tasks = ceildiv(m, 2 * kernel_shape.simd_rows)
    var max_col_tasks = ceildiv(n, kernel_shape.simd_cols * simd_size)
    num_tasks = min(num_tasks, max_row_tasks * max_col_tasks)

    return num_tasks


@value
struct SubMatmulConfig:
    """Static configuration of sub-matrices in parallel matmul."""

    # Starting Indices of sub-matrices.
    var offset: IndexList[3]

    # Dimension of sub-matrices.
    var shape: IndexList[3]

    @always_inline
    fn is_valid(self) -> Bool:
        return self.shape > Index(0, 0, 0)


# The work is first grouped into blocks for alignment and load/store efficiency.
# This will partition the work blocks between tasks as even as possible.
@always_inline
fn partition_work(
    task_id: Int, num_tasks: Int, work: Int, work_block_size: Int
) -> IndexList[2]:
    var num_work_blocks = ceildiv(work, work_block_size)
    var blocks_per_task = num_work_blocks // num_tasks
    var blocks_per_task_extra = num_work_blocks % num_tasks

    var work_per_task = blocks_per_task * work_block_size
    var work_id = (
        work_per_task * task_id + blocks_per_task_extra * work_block_size
    )

    if task_id < blocks_per_task_extra:
        work_per_task = (blocks_per_task + 1) * work_block_size
        work_id = task_id * work_per_task
        return IndexList[2](work_id, min(work - work_id, work_per_task))

    return IndexList[2](work_id, min(work - work_id, work_per_task))


fn get_partitioned_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    @parameter
    if use_i8mm:
        # i8mm needs to have even partitions in m.
        # Only the last range is allowed to be odd.
        var partition = get_partitioned_matmul_mojo[
            b_type, kernel_rows, kernel_cols, use_i8mm
        ](m // 2, n, k, task_id, num_tasks)
        var t0 = 2 * partition.offset[0]
        var t1 = 2 * partition.shape[0]
        if t0 + t1 == m - 1:
            t1 = m - t0
        partition.offset[0] = t0
        partition.shape[0] = t1
        return partition
    else:
        return get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
            m, n, k, task_id, num_tasks
        )


fn get_partitioned_matmul_mojo[
    b_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
    use_i8mm: Bool = False,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    var shape = get_partitioned_matmul_mojo_shape[
        b_type, kernel_rows, kernel_cols, use_i8mm
    ](m, n, k, num_tasks)
    var num_row_tasks = shape[0]
    var num_col_tasks = shape[1]
    var row_task_id = task_id // num_col_tasks
    var col_task_id = task_id % num_col_tasks

    var row_range = partition_work(row_task_id, num_row_tasks, m, kernel_rows)
    var col_range = partition_work(col_task_id, num_col_tasks, n, kernel_cols)
    return SubMatmulConfig(
        Index(row_range[0], col_range[0], 0),
        Index(row_range[1], col_range[1], k),
    )


fn get_partitioned_matmul_mojo_shape[
    b_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
    use_i8mm: Bool,
](m: Int, n: Int, k: Int, num_tasks: Int) -> IndexList[2]:
    var num_row_tasks = 1
    var num_col_tasks = 1

    var min_work = m * n

    var num_packs_m = ceildiv(m, kernel_rows)
    var num_packs_n = ceildiv(n, kernel_cols)
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
        var worki = kernel_cols * max((num_packs_n // num_tasks), 1)
        # ensure the work in m is not much smaller than in n
        var num_packs_m2 = ceildiv(m, align_down(worki, kernel_rows))
        if num_packs_n * num_packs_m2 >= num_tasks:
            max_num_packs_m = min(max_num_packs_m, num_packs_m2)

    max_num_packs_m = min(max_num_packs_m, num_tasks)
    max_num_packs_n = min(max_num_packs_n, num_tasks)
    # Loop over all possible partitions and find the the partition that balances the work best.
    for j in range(max_num_packs_m, 0, -1):
        var workj = kernel_rows * ceildiv(num_packs_m, j) if j != 1 else m
        for i in range(min(num_tasks // j, max_num_packs_n), 0, -1):
            var worki = kernel_cols * ceildiv(num_packs_n, i) if i != 1 else n
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
fn get_kernel_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    *,
    kernel_type: Bool = False,
]() -> KernelConfig:
    """Utility function to extract matmul configuration parameters for exported
    Functions.
        TODO: Add target dependent configuration parameters.
    """
    alias simd_size = simdwidthof[c_type]()

    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()

    return KernelConfig(
        kernel_rows=kernel_shape.simd_rows,
        kernel_cols=kernel_shape.simd_cols * simd_size,
        simd_size=simd_size,
        packed_shape=DimList.create_unknown[3](),
    )


@always_inline
fn use_vnni_fn[a_type: DType, b_type: DType, c_type: DType]() -> Bool:
    @parameter
    if has_neon_int8_dotprod() and not has_neon_int8_matmul():
        return (
            (a_type is DType.int8 and b_type is DType.int8)
            or (a_type is DType.uint8 and b_type is DType.uint8)
        ) and c_type is DType.int32
    elif has_avx2():
        return (
            a_type is DType.uint8
            and b_type is DType.int8
            and c_type is DType.int32
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
            (a_type is DType.uint8 and b_type is DType.uint8)
            or (a_type is DType.uint8 and b_type is DType.int8)
            or (a_type is DType.int8 and b_type is DType.int8)
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


fn dispatch_get_kernel_type[
    func: fn[x: Bool] () raises capturing [_] -> None,
](m: Int, n: Int, k: Int) raises:
    if get_kernel_type(m, n, k):
        func[True]()
    else:
        func[False]()


fn dispatch_get_kernel_type[
    func: fn[x: Bool] () capturing [_] -> None,
](m: Int, n: Int, k: Int):
    if get_kernel_type(m, n, k):
        func[True]()
    else:
        func[False]()


@always_inline
fn packA_i8mm[
    a_type: DType
](
    t0: Int,
    t1: Int,
    k: Int,
    a_ptr: UnsafePointer[Scalar[a_type]],
    a_packed_ptr: UnsafePointer[Scalar[a_type]],
):
    @always_inline
    @__copy_capture(k)
    @parameter
    fn packA_helper[nrow: Int](offset: Int):
        var kl = align_down(k, 8)
        var kh = align_up(k, 8)
        var j = t0 + offset
        for l in range(0, k, 8):

            @parameter
            for idx in range(nrow):
                var t0 = a_ptr.load[width=8]((j + idx) * k + l)
                a_packed_ptr.store(kh * j + 2 * l + 8 * idx, t0)

        @parameter
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


@value
@register_passable("trivial")
struct InnerKernelID:
    alias DEFAULT = InnerKernelID(0)
    alias VNNI = InnerKernelID(1)
    alias NEON = InnerKernelID(2)
    alias I8MM = InnerKernelID(3)

    var value: Int

    @always_inline
    fn __eq__(self, rhs: InnerKernelID) -> Bool:
        return self.value == rhs.value


@always_inline
fn select_inner_kernel[
    a_type: DType, b_type: DType, c_type: DType
]() -> InnerKernelID:
    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    @parameter
    if use_i8mm:
        return InnerKernelID.I8MM
    elif has_neon() and not use_vnni and not use_i8mm:
        return InnerKernelID.NEON
    elif not use_vnni and not has_neon():
        return InnerKernelID.DEFAULT
    else:
        return InnerKernelID.VNNI


@always_inline
fn apply_epilogue[
    elementwise_lambda: elementwise_epilogue_type,
    dst_layout: Layout,
    dst_element_layout: Layout = Layout(1, 1),
](src: LayoutTensor, offset: Int):  # register or shared memory
    # Check if input is 2D simd tile. This is only for double buffer gemm
    # TODO: extend it to 1D simd tile.
    @parameter
    if (
        src.element_layout.rank() == 2
        and dst_element_layout.shape == src.element_layout.shape
        and dst_element_layout.stride[1] == 1
        and src.element_layout.stride[1] == 1
    ):
        # update an element tensor.
        alias num_copies = src.element_layout.shape[0].value()
        alias vec_width = src.element_layout.shape[1].value()

        @parameter
        for i in range(dst_layout.size()):
            # Offset to the current element.
            alias src_offset = src.layout(i)
            alias dst_offset = dst_layout(i)

            @parameter
            for j in range(num_copies):
                alias src_idx = src_offset + src.element_layout(j)
                alias dst_idx = dst_offset + dst_element_layout(j)
                # C matrix dimension. For 2D simd tile, element_layout perserves
                # the matrix dimension, layout doesn't.
                alias N = dst_element_layout.stride[0].value()

                var vec = src.ptr.load[
                    width=vec_width,
                    alignment = alignof[SIMD[src.dtype, vec_width]](),
                ](src_idx)

                var m = (dst_idx + offset) // N
                var n = (dst_idx + offset) % N

                elementwise_lambda[src.dtype, vec_width]((m, n), vec)

    # Scalar case
    # TODO: 1D vector is included, should handle it in a separate branch.
    else:
        constrained[dst_element_layout.rank() == 1]()

        @parameter
        for i in range(src.layout.size() * src.element_size):
            alias src_idx = make_layout(src.element_layout, src.layout)(i)
            alias dst_idx = make_layout(dst_element_layout, dst_layout)(i)
            # C matrix dimension. For scalar or 1D vector element, the layout
            # preserves the matrix dimension.
            alias N = dst_layout.stride[0].value()

            var m = (src_idx + offset) // N
            var n = (src_idx + offset) % N

            elementwise_lambda[src.dtype, 1]((m, n), src.ptr[src_idx + offset])
