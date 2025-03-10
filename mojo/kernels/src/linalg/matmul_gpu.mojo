# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import InlineArray, OptionalReg
from math import align_down, align_up, ceildiv
from pathlib import Path
from sys import (
    alignof,
    bitwidthof,
    env_get_bool,
    env_get_int,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_defined,
    llvm_intrinsic,
    simdwidthof,
)

from algorithm.functional import elementwise, tile_and_unswitch
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute, LaunchAttribute
from gpu.host._compile import _get_gpu_target
from gpu.host.info import A100, H100, DEFAULT_GPU_ARCH
from gpu.memory import AddressSpace, CacheOperation, load
from gpu.mma import ld_matrix, mma
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
    from_ndbuffer_row_major,
    vectorize,
)
from memory import UnsafePointer, bitcast, memset_zero, stack_allocation

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ._multistage_gemm_gpu import (
    multistage_gemm_kernel,
    multistage_gemm_split_k_kernel,
)
from .matmul_sm90 import (
    hopper_matmul_tma_wgmma,
    warp_specialize_gemm_with_multicasting,
)
from .gemv import gemv_gpu
from .utils import GemmShape, apply_epilogue, elementwise_epilogue_type
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    _get_block_warp_tile_shape,
    select_config,
)
from linalg.matmul_tile_scheduler import MatmulSchedule
from ._amd_gemm_gpu import gemm_kernel as amd_gemm_kernel

alias tile_shapes_64X64X32 = _get_block_warp_tile_shape[64, 64, 32]()
alias tile_shapes_64X128X32 = _get_block_warp_tile_shape[64, 128, 32]()
alias tile_shapes_64X256X32 = _get_block_warp_tile_shape[64, 256, 32]()
alias tile_shapes_128X64X32 = _get_block_warp_tile_shape[128, 64, 32]()
alias tile_shapes_128X128X32 = _get_block_warp_tile_shape[128, 128, 32]()
alias tile_shapes_128X256X32 = _get_block_warp_tile_shape[128, 256, 32]()
alias tile_shapes_256X64X32 = _get_block_warp_tile_shape[256, 64, 32]()
alias tile_shapes_256X128X32 = _get_block_warp_tile_shape[256, 128, 32]()
alias tile_shapes_256X256X32 = _get_block_warp_tile_shape[256, 256, 32]()
alias tile_shapes_64X64X64 = _get_block_warp_tile_shape[64, 64, 64]()
alias tile_shapes_64X128X64 = _get_block_warp_tile_shape[64, 128, 64]()
alias tile_shapes_64X256X64 = _get_block_warp_tile_shape[64, 256, 64]()
alias tile_shapes_128X64X64 = _get_block_warp_tile_shape[128, 64, 64]()
alias tile_shapes_128X128X64 = _get_block_warp_tile_shape[128, 128, 64]()
alias tile_shapes_128X256X64 = _get_block_warp_tile_shape[128, 256, 64]()
alias tile_shapes_256X64X64 = _get_block_warp_tile_shape[256, 64, 64]()
alias tile_shapes_256X128X64 = _get_block_warp_tile_shape[256, 128, 64]()
alias tile_shapes_256X256X64 = _get_block_warp_tile_shape[256, 256, 64]()


@always_inline
fn __nvvm_ldg_f4[type: DType](x: UnsafePointer[Scalar[type]]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type == DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[Float32](), alignment)
        )
    elif type == DType.bfloat16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[BFloat16](), alignment)
        )
    elif type == DType.float16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f16.p0v4f16",
                SIMD[DType.float16, 4],
            ](x.bitcast[Float16](), alignment)
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
    var col = global_idx.x
    var row = global_idx.y

    # Local index in the c sub-matrix updated by current block.
    var localCol = thread_idx.x
    var localRow = thread_idx.y

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
    var x = global_idx.x
    var y = global_idx.y

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
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
](
    c: NDBuffer[c_type, 2, _],
    a: NDBuffer[a_type, 2, _],
    b: NDBuffer[b_type, 2, _],
    ctx: DeviceContext,
) raises:
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
    var multi_gemm_cond = (
        (m > 1 or (has_amd_gpu_accelerator() and transpose_b == False))
        and n % 128 == 0
        and k % 32 == 0
        and k >= 128
    )
    # fmt: off
    # Require Static K, N in A, B, C
    alias multistage_gemm_supported_shape = b_shape.all_known[2]() \
                                        and a_shape.has_value[1]() \
                                        and c_shape.has_value[1]()
    # fmt: on

    @parameter
    if (
        matmul_supported_format
        and (has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator())
        and use_tensor_core
        and multistage_gemm_supported_shape
    ):
        if multi_gemm_cond:
            alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

            @parameter
            if has_amd_gpu_accelerator():
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.mi300x_128x128_1 if transpose_b else kernels.mi300x_128x128_2,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    kernels.mi300x_128x128_1 if transpose_b else kernels.mi300x_128x128_2,
                    ctx,
                )
                return

            @parameter
            if is_defined["AUTOTUNING_MODE"]():
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.tuning_config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    kernels.tuning_config,
                    ctx,
                )
                return

            # Allow caller to overwrite dispatch heuristic with their own config.
            @parameter
            if config:
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = config.value(),
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    config.value(),
                    ctx,
                )
                return

            alias use_A100_kernels_on_H100 = env_get_int[
                "USE_EXPERIMENTAL_KERNELS", 0
            ]()

            @parameter
            if (
                a_type == b_type
                and a_type.is_half_float()
                and ctx.device_info is H100
                and transpose_b
                and not use_A100_kernels_on_H100
            ):
                alias static_N = c_shape.get[1]()
                alias static_K = a_shape.get[1]()

                # GTC matmul configs
                @parameter
                if static_N == 2560 and static_K == 8192:
                    if m == 8192:
                        alias M8192_N2560_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M8192_N2560_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                    elif m == 4096:
                        alias M4096_N2560_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M4096_N2560_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 8192 and static_K == 2048:
                    if m == 8192:
                        alias M8192_N8192_K2048_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M8192_N8192_K2048_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                    elif m == 4096:
                        alias M4096_N8192_K2048_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M4096_N8192_K2048_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 14336 and static_K == 8192:
                    if m == 8192:
                        alias M8192_N14336_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M8192_N14336_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                    elif m == 4096:
                        alias M4096_N14336_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M4096_N14336_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 8192 and static_K == 7168:
                    if m == 8192:
                        alias M8192_N8192_K7168_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M8192_N8192_K7168_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                    elif m == 4096:
                        alias M4096_N8192_K7168_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256, 16),
                        ](
                            block_tile_shape=Index(128, 256, 64),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            config=M4096_N8192_K7168_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                alias default_config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 256, 16),
                ](
                    block_tile_shape=Index(128, 256, 64),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    config=default_config,
                    schedule = MatmulSchedule.NONE,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

            alias use_A100_kenrels = ctx.device_info is A100 or (
                ctx.device_info is H100 and use_A100_kernels_on_H100 != 0
            )

            @parameter
            if (
                a_type == b_type
                and a_type.is_half_float()
                and use_A100_kenrels
                and transpose_b
            ):
                alias static_K = a_shape.get[1]()
                alias static_N = c_shape.get[1]()

                @parameter
                if static_K == 4096 and static_N == 4096:
                    if m <= 16:
                        alias M16_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(16, 64, 128),
                            warp_tile_shape=Index(16, 32, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M16_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M16_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 32:
                        alias M32_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(32, 64, 64),
                            warp_tile_shape=Index(16, 64, 32),
                            num_pipeline_stages=6,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M32_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M32_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 64:
                        alias M64_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 64, 64),
                            warp_tile_shape=Index(32, 64, 32),
                            num_pipeline_stages=5,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M64_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M64_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 128:
                        alias M128_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 128, 32),
                            warp_tile_shape=Index(64, 64, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M128_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M128_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 128 < m <= 256:
                        alias M256_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_64X256X32[0],
                            warp_tile_shape=tile_shapes_64X256X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M256_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M256_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 256 < m <= 512:
                        alias M512_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M512_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M512_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 512 < m <= 768:
                        alias M768_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 768 < m <= 1280:
                        alias tile_shapes = _get_block_warp_tile_shape[
                            128, 128, 32
                        ]()
                        alias M896_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M896_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M896_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 1280 < m <= 1606:
                        alias M1606_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1606_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1606_N4096_K4096_config,
                            ctx,
                        )
                        return
                    if 1606 < m <= 2048:
                        alias tile_shapes = _get_block_warp_tile_shape[
                            128, 128, 32
                        ]()
                        alias M2048_N4096_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M2048_N4096_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M2048_N4096_K4096_config,
                            ctx,
                        )
                        return

                @parameter
                if static_K == 4096 and static_N == 14336:
                    if m <= 16:
                        alias M16_N14336_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(16, 64, 128),
                            warp_tile_shape=Index(16, 32, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M16_N14336_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M16_N14336_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 32:
                        alias M32_N14336_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(32, 64, 64),
                            warp_tile_shape=Index(16, 64, 32),
                            num_pipeline_stages=6,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M32_N14336_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M32_N14336_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 64:
                        alias M64_N14336_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 64, 64),
                            warp_tile_shape=Index(32, 64, 32),
                            num_pipeline_stages=5,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M64_N14336_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M64_N14336_K4096_config,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 4096 and static_K == 14336:
                    if m <= 16:
                        alias M16_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(16, 64, 128),
                            warp_tile_shape=Index(16, 32, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M16_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M16_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 32:
                        alias M32_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(32, 64, 64),
                            warp_tile_shape=Index(16, 64, 32),
                            num_pipeline_stages=6,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M32_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M32_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 64:
                        alias M64_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 64, 64),
                            warp_tile_shape=Index(32, 64, 32),
                            num_pipeline_stages=5,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M64_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M64_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 128:
                        alias M128_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_64X256X64[0],
                            warp_tile_shape=tile_shapes_64X256X64[1],
                            num_pipeline_stages=4,
                            num_k_partitions=3,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M128_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M128_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 512:
                        alias M512_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=3,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M512_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M512_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 768:
                        alias M768_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 896:
                        alias M896_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X64[0],
                            warp_tile_shape=tile_shapes_128X128X64[1],
                            num_pipeline_stages=4,
                            num_k_partitions=3,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M896_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M896_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 1024:
                        alias M1024_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1024_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1024_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 1152:
                        alias M1152_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=3,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1152_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1152_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 1280:
                        alias M1280_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1280_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1280_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 1606:
                        alias M1606_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1606_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1606_N4096_K14336_config,
                            ctx,
                        )
                        return
                    if m <= 2048:
                        alias M2048_N4096_K14336_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M2048_N4096_K14336_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M2048_N4096_K14336_config,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 128256 and static_K == 4096:
                    if m <= 128:
                        alias M128_N128256_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M128_N128256_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M128_N128256_K4096_config,
                            ctx,
                        )
                        return
                    if 768 < m <= 896:
                        alias M768_N128256_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X64[0],
                            warp_tile_shape=tile_shapes_128X128X64[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N128256_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N128256_K4096_config,
                            ctx,
                        )
                        return
                    if 896 < m <= 2048 or m <= 768:
                        alias M2048_N128256_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M2048_N128256_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M2048_N128256_K4096_config,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 28672 and static_K == 4096:
                    if m <= 16:
                        alias M16_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(16, 64, 128),
                            warp_tile_shape=Index(16, 32, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M16_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M16_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 32:
                        alias M32_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(32, 64, 64),
                            warp_tile_shape=Index(16, 64, 32),
                            num_pipeline_stages=6,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M32_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M32_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 64:
                        alias M64_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 64, 64),
                            warp_tile_shape=Index(32, 64, 32),
                            num_pipeline_stages=5,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M64_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M64_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 128:
                        alias M128_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M128_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M128_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 256:
                        alias M256_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_64X256X64[0],
                            warp_tile_shape=tile_shapes_64X256X64[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M256_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M256_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 512:
                        alias M512_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M512_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M512_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 768:
                        alias M768_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 896:
                        alias M896_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M896_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M896_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 1024:
                        alias M1024_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1024_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1024_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 1152:
                        alias M1152_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1152_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1152_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 1280:
                        alias M1280_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1280_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1280_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 1606:
                        alias M1606_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1606_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1606_N28672_K4096_config,
                            ctx,
                        )
                        return
                    if m <= 2048:
                        alias M2048_N28672_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M2048_N28672_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M2048_N28672_K4096_config,
                            ctx,
                        )
                        return

                @parameter
                if static_N == 6144 and static_K == 4096:
                    if m <= 16:
                        alias M16_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(16, 64, 128),
                            warp_tile_shape=Index(16, 32, 32),
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M16_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M16_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 16 < m <= 32:
                        alias M32_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(32, 64, 64),
                            warp_tile_shape=Index(16, 64, 32),
                            num_pipeline_stages=6,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M32_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M32_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 32 < m <= 64:
                        alias M64_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 64, 64),
                            warp_tile_shape=Index(32, 64, 32),
                            num_pipeline_stages=5,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M64_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M64_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 64 < m <= 128:
                        alias M128_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=Index(64, 128, 64),
                            warp_tile_shape=Index(64, 64, 32),
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                            num_warp_k_partitions=2,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M128_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M128_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 128 < m <= 256:
                        alias M256_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X64[0],
                            warp_tile_shape=tile_shapes_128X128X64[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M256_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M256_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 256 < m <= 512:
                        alias M512_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M512_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M512_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 512 < m <= 768:
                        alias M768_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 768 < m <= 896:
                        alias M896_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X32[0],
                            warp_tile_shape=tile_shapes_128X256X32[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M896_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M896_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 896 < m <= 1024:
                        alias M1024_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1024_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1024_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 1024 < m <= 1152:
                        alias M768_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 1152 < m <= 1280:
                        alias M768_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X128X32[0],
                            warp_tile_shape=tile_shapes_128X128X32[1],
                            num_pipeline_stages=4,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M768_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M768_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 1280 < m <= 1606:
                        alias M1606_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_128X256X64[0],
                            warp_tile_shape=tile_shapes_128X256X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M1606_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M1606_N6144_K4096_config,
                            ctx,
                        )
                        return
                    if 1606 < m <= 2048:
                        alias M2048_N6144_K4096_config = MatmulConfig[
                            a_type, b_type, c_type, transpose_b
                        ](
                            block_tile_shape=tile_shapes_256X128X64[0],
                            warp_tile_shape=tile_shapes_256X128X64[1],
                            num_pipeline_stages=3,
                            num_k_partitions=1,
                        )
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=M2048_N6144_K4096_config,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                        ](
                            rebind[NDBuffer[c_type, 2, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b_shape]](b),
                            M2048_N6144_K4096_config,
                            ctx,
                        )
                        return

            var best_config = select_config[
                a_type, b_type, c_type, transpose_b
            ](m, n, k, ctx)

            if best_config == kernels.ampere_256x64_4:
                alias config = kernels.ampere_256x64_4
                multistage_gemm[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    best_config,
                    ctx,
                )

            elif best_config == kernels.ampere_256x128_3:
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.ampere_256x128_3,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    best_config,
                    ctx,
                )

            else:  # Default kernel 128x128_4
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.ampere_128x128_4,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](
                    rebind[NDBuffer[c_type, 2, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b_shape]](b),
                    best_config,
                    ctx,
                )

            return

    if n == 1 or m == 1:
        gemv_gpu[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, ctx)
        return

    alias BLOCK_DIM = 16
    ctx.enqueue_function[
        matmul_kernel_naive[
            c_type,
            a_type,
            b_type,
            BLOCK_DIM,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
    ](
        c.data,
        a.data,
        b.data,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )


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
) raises:
    alias simd_width = simdwidthof[c_type, target = _get_gpu_target()]()
    var num_partitions = work_space.dim[0]()
    var M = c.dim[0]()
    var N = c.dim[1]()

    @always_inline
    @__copy_capture(c, work_space, num_partitions)
    @parameter
    fn _reduce[simd_width: Int, rank: Int](c_coord: IndexList[rank]):
        var idx = Index(0, c_coord[0], c_coord[1])
        var vec = work_space.load[width=simd_width](idx)
        for k in range(1, num_partitions):
            vec += work_space.load[width=simd_width](
                Index(k, c_coord[0], c_coord[1])
            )

        alias alignment = alignof[SIMD[c_type, simd_width]]()

        @parameter
        if elementwise_lambda_fn:
            alias epilogue = elementwise_lambda_fn.value()
            epilogue[alignment=alignment](
                rebind[IndexList[2]](c_coord), vec.cast[c_type]()
            )
        else:
            c.store[width=simd_width, alignment=alignment](
                rebind[IndexList[2]](c_coord), vec.cast[c_type]()
            )

    elementwise[_reduce, simd_width, target="gpu"](Index(M, N), ctx)


fn multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    serial_reduction: Bool = False,
](
    c: NDBuffer[c_type, 2, c_shape],
    a: NDBuffer[a_type, 2, a_shape],
    b: NDBuffer[b_type, 2, b_shape],
    runtime_config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()

    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b = from_ndbuffer_row_major(b)

    if runtime_config.num_k_partitions > 1:

        @parameter
        if serial_reduction:
            constrained[
                c_type == DType.bfloat16,
                "serial reduction is unsupported for this config",
            ]()
            alias work_space_type = config.split_k_reduction_type

            # For the serial reduction we dont use workspace
            var work_space = NDBuffer[work_space_type, 3](
                UnsafePointer[Scalar[work_space_type]](),
                Index(0, 0, 0),
            )

            alias BM = config.block_tile_shape[0]
            alias BN = config.block_tile_shape[1]

            var locks_data = ctx.enqueue_create_buffer[DType.int32](
                ceildiv(M, BM) * ceildiv(N, BN)
            )
            ctx.enqueue_memset(locks_data, 0)

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
                serial_reduction,
            ]

            ctx.enqueue_function[gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                work_space,
                runtime_config.num_k_partitions,
                locks_data.unsafe_ptr(),
                grid_dim=runtime_config.grid_dim(M, N),
                block_dim=runtime_config.block_dim(),
                shared_mem_bytes=runtime_config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    config.shared_mem_usage()
                ),
            )

            _ = locks_data^
            return

        else:
            alias work_space_type = config.split_k_reduction_type
            var work_space_data = ctx.enqueue_create_buffer[work_space_type](
                runtime_config.num_k_partitions * M * N
            )
            var work_space = NDBuffer[work_space_type, 3](
                work_space_data.unsafe_ptr(),
                Index(Int(runtime_config.num_k_partitions), M, N),
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
                False,
            ]

            ctx.enqueue_function[gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                work_space,
                runtime_config.num_k_partitions,
                UnsafePointer[Int32](),
                grid_dim=runtime_config.grid_dim(M, N),
                block_dim=runtime_config.block_dim(),
                shared_mem_bytes=runtime_config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    config.shared_mem_usage()
                ),
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
    @parameter
    if has_amd_gpu_accelerator() and transpose_b:
        alias gemm_kernel_type = amd_gemm_kernel[
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
        ctx.enqueue_function[gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            grid_dim=runtime_config.grid_dim(M, N),
            block_dim=runtime_config.block_dim(),
        )

    else:
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

        ctx.enqueue_function[gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            UnsafePointer[Int32](),
            grid_dim=runtime_config.grid_dim(M, N),
            block_dim=runtime_config.block_dim(),
            shared_mem_bytes=runtime_config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                config.shared_mem_usage()
            ),
        )
