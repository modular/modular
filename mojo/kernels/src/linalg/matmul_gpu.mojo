# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import InlineArray, OptionalReg
from math import align_down, align_up, ceildiv
from os import abort
from sys import alignof, llvm_intrinsic, simdwidthof, bitwidthof

from algorithm.functional import elementwise, tile_and_unswitch
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, BlockDim, BlockIdx, ThreadIdx, barrier, lane_id
from gpu.host import (
    DeviceContext,
    FuncAttribute,
    LaunchAttribute,
    AccessPolicyWindow,
    AccessProperty,
)
from gpu.host._compile import _get_nvptx_target
from gpu.memory import (
    AddressSpace,
    CacheOperation,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
    load,
)
from gpu.mma import ld_matrix, mma
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_local,
)
from layout.math import outer_product_acc
from layout.nd_buffer_stub import (
    copy_from_nd_buffer,
    distribute,
    vectorize,
    from_ndbuffer_row_major,
)
from memory import UnsafePointer, bitcast, stack_allocation, memset_zero

from utils import StaticIntTuple
from utils.index import Index
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ._multistage_gemm_gpu import (
    multistage_gemm_kernel,
    multistage_gemm_split_k_kernel,
)
from .utils import GemmShape, apply_epilogue, elementwise_epilogue_type
from .gemv import gemv_gpu
from .utils_gpu import MatmulKernels, MatmulConfig, select_config


@always_inline
fn __nvvm_ldg_f4[type: DType](x: UnsafePointer[Scalar[type]]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type == DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[DType.float32](), alignment)
        )
    elif type == DType.bfloat16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[DType.bfloat16](), alignment)
        )
    elif type == DType.float16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f16.p0v4f16",
                SIMD[DType.float16, 4],
            ](x.bitcast[DType.float16](), alignment)
        )
    else:
        constrained[False, "Unhandled DType"]()
        return 0


fn matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c_ptr: UnsafePointer[Scalar[c_type]],
    a_ptr: UnsafePointer[Scalar[a_type]],
    b_ptr: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    """Matrix Multiplication using shared memory.
    This version loads blocks of size tile_size x tile_size from A and B
    and updates a tile_size x tile_size in C.
    The thread block should have shape (tile_size, tile_size, 1). Each
    thread is mapped one element in C. The grid should have shape
    (N/tile_size, M/tile_size, 1). N is the first dimension for coalesced
    access.
    """
    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    # Allocate A, B tile in shared memory.
    var a_shared = stack_allocation[
        tile_size * tile_size,
        a_type,
        address_space = AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        b_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Global index in C.
    # These are the same indices in A and B when loading to SRAM.
    # Map thread x to column for coalesced access in B.
    var col = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var row = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    # Local index in the c sub-matrix updated by current block.
    var localCol = ThreadIdx.x()
    var localRow = ThreadIdx.y()

    # Result of current thread in C.
    var result = Scalar[s_type](0)

    var K_roundbytile = align_down(k, tile_size)
    # Can't use 0 as tile size so set to 1 when the remainder is 0.
    var K_remainder = k - K_roundbytile if k - K_roundbytile > 0 else 1

    @parameter
    @__copy_capture(row, localCol, a, b, localRow, col, a_shared, b_shared)
    @always_inline
    fn update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: Scalar[a_type]

        @parameter
        if not full_tile:
            a_val = a[row, offset + localCol] if (
                row < m and offset + localCol < k
            ) else 0.0
        else:
            a_val = a[row, offset + localCol] if row < m else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Scalar[b_type]

        @parameter
        if not full_tile:
            b_val = b[offset + localRow, col] if (
                col < n and offset + localRow < k
            ) else 0.0
        else:
            b_val = b[offset + localRow, col] if col < n else 0.0
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for kk in range(tile_size):
            result += (
                a_shared[localRow * tile_size + kk].cast[s_type]()
                * b_shared[kk * tile_size + localCol].cast[s_type]()
            )

        barrier()

    tile_and_unswitch[update_tile](
        0, k, VariadicList[Int](tile_size, K_remainder)
    )

    if row < m and col < n:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(row, col), result.cast[c_type]()
            )
        else:
            c[Index(row, col)] = result.cast[c_type]()


fn matmul_kernel_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c_ptr: UnsafePointer[Scalar[c_type]],
    a_ptr: UnsafePointer[Scalar[a_type]],
    b_ptr: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    var x = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var y = BlockIdx.y() * BlockDim.y() + ThreadIdx.y()

    if x >= m or y >= n:
        return

    var a = NDBuffer[a_type, 2](a_ptr, Index(m, k))
    var accum = Scalar[s_type]()

    @parameter
    if transpose_b:
        var b = NDBuffer[b_type, 2](b_ptr, Index(n, k))
        for i in range(k):
            accum = a[x, i].cast[s_type]() * b[y, i].cast[s_type]() + accum

    else:
        var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
        for i in range(k):
            accum = a[x, i].cast[s_type]() * b[i, y].cast[s_type]() + accum

    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[Index(x, y)] = accum.cast[c_type]()


@always_inline
fn _matmul_gpu[
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    single_thread_blocking_override: Bool = False,
](
    c: NDBuffer[_, 2, _],
    a: NDBuffer[_, 2, _],
    b: NDBuffer[_, 2, _],
    ctx: DeviceContext,
):
    # HACK HACK HACK https://github.com/modularml/modular/issues/22959
    # single_thread_blocking_override should not be allowed, but the graph
    # compiler has a special case that does not insert the
    # on the GPU
    # constrained[
    #     not single_thread_blocking_override,
    #     "single_thread_blocking_override not applicable",
    # ]()
    try:
        alias a_type = a.type
        alias b_type = b.type
        alias c_type = c.type
        alias a_shape = a.shape
        alias b_shape = b.shape
        alias c_shape = c.shape
        var shape = GemmShape.get[transpose_b=False](c, a, b)
        var m = shape.M
        var n = shape.N
        var k = shape.K
        alias s_type = DType.float32 if (
            a_type == DType.bfloat16 or a_type == DType.float16
        ) else c_type
        alias matmul_supported_format = (
            a_type in (DType.float32, DType.bfloat16)
            and b_type in (DType.float32, DType.bfloat16)
            and c_type in (DType.float32, DType.bfloat16)
        )
        # NOTE: k has to be a multiple of BK * num_stages. Hard coded this condition to 128 for now.
        # TODO: Need to find a better dispatch strategy.
        var multi_gemm_cond = (m > 1 and n % 128 == 0 and k % 128 == 0)
        # fmt: off
        # Require Static K, N in A, B, C
        alias multistage_gemm_supported_shape = b_shape.all_known[2]() \
        and a_shape.has_value[1]() \
        and c_shape.has_value[1]()
        # fmt: on

        @parameter
        if (
            matmul_supported_format
            and use_tensor_core
            and multistage_gemm_supported_shape
        ):
            if multi_gemm_cond:
                alias kernels = MatmulKernels[
                    a_type, b_type, c_type, transpose_b
                ]()

                var best_config = select_config[
                    a_type, b_type, c_type, transpose_b
                ](m, n, k)

                if best_config == kernels.ampere_256x64_4:
                    alias config = kernels.ampere_256x64_4
                    multistage_gemm[
                        c_type,
                        c_shape,
                        a_type,
                        a_shape,
                        b_type,
                        b_shape,
                        transpose_b,
                        config,
                        elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b_shape]](b),
                        best_config,
                        ctx,
                    )

                else:  # Default kernel 128x128_4
                    alias config = kernels.ampere_128x128_4
                    multistage_gemm[
                        c_type,
                        c_shape,
                        a_type,
                        a_shape,
                        b_type,
                        b_shape,
                        transpose_b,
                        config,
                        elementwise_lambda_fn,
                    ](
                        rebind[NDBuffer[c_type, 2, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b_shape]](b),
                        best_config,
                        ctx,
                    )

                return

        if m == 1 or n == 1:
            gemv_gpu[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](c, a, b, ctx)
            return

        alias BLOCK_DIM = 16
        var gpu_func = ctx.compile_function[
            matmul_kernel_naive[
                a_type,
                b_type,
                c_type,
                BLOCK_DIM,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
        ]()
        ctx.enqueue_function(
            gpu_func,
            c.data,
            a.data,
            b.data,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    except e:
        abort(e)


@always_inline
fn split_k_reduce[
    type: DType,
    a_shape: DimList,
    b_shape: DimList,
](
    A: NDBuffer[type, 2, a_shape],
    B: NDBuffer[type, 2, b_shape],
    num_partition: UInt,
    ctx: DeviceContext,
):
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()
    var M = A.dim[0]()
    var N = A.dim[1]()

    @always_inline
    @__copy_capture(A, B, N)
    @parameter
    fn _reduce[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[2]](idx0)
        var i = idx[0]
        var j = idx[1]

        var vec = A.load[width=simd_width](idx)
        for k in range(num_partition):
            vec += B.load[width=simd_width](i, j + k * N)
        A.store[width=simd_width](idx, vec)

    elementwise[_reduce, pack_size, target="cuda"](StaticIntTuple[2](M, N), ctx)


@always_inline
fn split_k_reduce[
    c_type: DType,
    work_space_type: DType,
    c_shape: DimList,
    work_space_shape: DimList,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    work_space: NDBuffer[work_space_type, 3, work_space_shape],
    ctx: DeviceContext,
):
    alias simd_width = simdwidthof[c_type, target = _get_nvptx_target()]()
    var num_partitions = work_space.dim[0]()
    var M = c.dim[0]()
    var N = c.dim[1]()

    @always_inline
    @__copy_capture(c, work_space, num_partitions)
    @parameter
    fn _reduce[simd_width: Int, rank: Int](c_coord: StaticIntTuple[rank]):
        var idx = Index(0, c_coord[0], c_coord[1])
        var vec = work_space.load[width=simd_width](idx)
        for k in range(1, num_partitions):
            vec += work_space.load[width=simd_width](
                Index(k, c_coord[0], c_coord[1])
            )

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            epilogue(rebind[StaticIntTuple[2]](c_coord), vec.cast[c_type]())
        else:
            c.store[width=simd_width](
                rebind[StaticIntTuple[2]](c_coord), vec.cast[c_type]()
            )

    elementwise[_reduce, simd_width, target="cuda"](Index(M, N), ctx)


def multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    runtime_config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    ctx: DeviceContext,
):
    var M = c.dim[0]()
    var N = c.dim[1]()

    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b = from_ndbuffer_row_major(b)

    # Split K dispatch
    if runtime_config.num_k_partitions > 1:
        alias work_space_type = config.split_k_reduction_type
        var work_space_data = ctx.create_buffer[work_space_type](
            runtime_config.num_k_partitions * M * N
        )
        var work_space = NDBuffer[work_space_type, 3](
            work_space_data.ptr,
            Index(int(runtime_config.num_k_partitions), M, N),
        )

        alias gemm_kernel_type = multistage_gemm_split_k_kernel[
            c_type,
            tensor_c.layout,
            a_type,
            tensor_a.layout,
            b_type,
            tensor_b.layout,
            work_space_type,
            transpose_b,
            config,
            elementwise_lambda_fn,
        ]

        var gemm_kernel = ctx.compile_function[
            gemm_kernel_type,
            # dump_ptx=Path("./pipeline-gemm.ptx"),
        ](
            threads_per_block=int(config.num_threads()),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                config.shared_mem_usage()
            ),
        )

        ctx.enqueue_function(
            gemm_kernel,
            tensor_c,
            tensor_a,
            tensor_b,
            work_space,
            runtime_config.num_k_partitions,
            grid_dim=runtime_config.grid_dim(M, N),
            block_dim=runtime_config.block_dim(),
            shared_mem_bytes=runtime_config.shared_mem_usage(),
        )

        split_k_reduce[
            c_type,
            work_space_type,
            c_shape,
            work_space.shape,
            elementwise_lambda_fn,
        ](c, work_space, ctx)

        _ = work_space_data^
        return

    # Dispatch w/o split K
    alias gemm_kernel_type = multistage_gemm_kernel[
        c_type,
        tensor_c.layout,
        a_type,
        tensor_a.layout,
        b_type,
        tensor_b.layout,
        transpose_b,
        config,
        elementwise_lambda_fn,
    ]

    var gemm_kernel = ctx.compile_function[gemm_kernel_type,](
        threads_per_block=int(config.num_threads()),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    ctx.enqueue_function(
        gemm_kernel,
        tensor_c,
        tensor_a,
        tensor_b,
        grid_dim=runtime_config.grid_dim(M, N),
        block_dim=runtime_config.block_dim(),
        shared_mem_bytes=runtime_config.shared_mem_usage(),
    )
