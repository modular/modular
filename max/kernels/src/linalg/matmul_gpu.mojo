# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from collections import OptionalReg
from math import align_down, ceildiv
from sys import (
    alignof,
    bitwidthof,
    env_get_bool,
    env_get_int,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_defined,
    llvm_intrinsic,
    simdwidthof,
)
from sys import sizeof
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
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.compile import get_gpu_target
from gpu.host.info import A100, H100
from gpu.memory import AddressSpace
from layout._ndbuffer_stub import (
    copy_from_nd_buffer,
    distribute,
    from_ndbuffer_row_major,
    vectorize,
)
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_local,
)
from linalg.matmul_tile_scheduler import MatmulSchedule
from memory import bitcast, stack_allocation

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type

from ._amd_gemm_gpu import gemm_kernel as amd_gemm_kernel
from ._multistage_gemm_gpu import (
    multistage_gemm_kernel,
    multistage_gemm_split_k_kernel,
)
from .dispatch_table_a100_gpu import create_matmul_configs_ampere
from .gemv import gemv_gpu
from .matmul_sm90 import (
    hopper_matmul_tma_wgmma,
    warp_specialize_gemm_with_multicasting,
)
from .matmul_vendor import matmul as matmul_vendor
from .utils import (
    GemmShape,
    apply_epilogue,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    _get_block_warp_tile_shape,
    select_config,
)


fn _find_largest_bn_for_sm90_matmul[dtype: DType, N: Int]() -> Int:
    @parameter
    if N % 8 != 0:
        return -1

    fn _get_max_bn() capturing -> Int:
        # For float8_e4m3fn maximum BN that will not result in register spilling is 160
        var BN = 160 if dtype == DType.float8_e4m3fn else 256
        while BN >= 8:
            if N % BN == 0:
                return BN
            BN -= 8
        return 8

    return _get_max_bn()


@always_inline
fn __nvvm_ldg_f4[type: DType](x: UnsafePointer[Scalar[type]]) -> SIMD[type, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[type, 4]]())

    @parameter
    if type is DType.float32:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[Float32](), alignment)
        )
    elif type is DType.bfloat16:
        return bitcast[type, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[BFloat16](), alignment)
        )
    elif type is DType.float16:
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
            accum += a[x, i].cast[s_type]() * b[y, i].cast[s_type]()

    else:
        var b = NDBuffer[b_type, 2](b_ptr, Index(k, n))
        for i in range(k):
            accum += a[x, i].cast[s_type]() * b[i, y].cast[s_type]()

    var c = NDBuffer[c_type, 2](c_ptr, Index(m, n))

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[Index(x, y)] = accum.cast[c_type]()


@fieldwise_init
@register_passable("trivial")
struct AMDSchedulerTuning(Copyable, Movable):
    var block_shape: IndexList[2]
    var tuning_values: IndexList[3]


@always_inline
fn _matmul_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
    _trace_description: StaticString = "",
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    alias K = a.shape.get[1]()
    alias a_shape = a.shape
    alias b_shape = b.shape
    alias c_shape = c.shape
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    try:
        return matmul_vendor[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            config=config,
            _trace_description=_trace_description,
        ](c, a, b, ctx)
    except:
        # fallback to multistage/naive gemms if the cublas failed. This is a workaround for now for KERN-1812
        @parameter
        if K * sizeof[a_type]() >= 8 * 16:
            alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()
            alias config = kernels.ampere_256x64_4
            multistage_gemm[
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                rebind[NDBuffer[c_type, 2, c.origin, c.shape]](c),
                rebind[NDBuffer[a_type, 2, a.origin, a.shape]](a),
                rebind[NDBuffer[b_type, 2, b.origin, b.shape]](b),
                config,
                ctx,
            )
        else:
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
        return


@always_inline
fn _matmul_gpu[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
    _trace_description: StaticString = "",
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
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
        a_type is DType.bfloat16 or a_type is DType.float16
    ) else c_type

    alias matmul_supported_format_nvidia = (
        a_type in (DType.float32, DType.bfloat16)
        and b_type in (DType.float32, DType.bfloat16)
        and c_type in (DType.float32, DType.bfloat16)
    )

    alias matmul_supported_format_amd = (
        a_type is DType.bfloat16
        and b_type is DType.bfloat16
        and c_type is DType.bfloat16
    )

    alias matmul_supported_format = matmul_supported_format_amd if has_amd_gpu_accelerator() else matmul_supported_format_nvidia

    # Only the H100 version of gemm supports the compute lambda
    # For the other kernels we wrap it around an epilogue lambda instead.
    @parameter
    @always_inline
    fn compute_lambda_wrapper[
        _type: DType, _width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_type, _width]):
        @parameter
        if elementwise_compute_lambda_fn:
            alias compute_lambda = elementwise_compute_lambda_fn.value()
            var output = compute_lambda(coords, val)
            c.store[alignment=alignment](
                coords, rebind[SIMD[c.type, _width]](output)
            )

    alias elementwise_lambda_wrapper = OptionalReg[elementwise_epilogue_type](
        compute_lambda_wrapper
    ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

    # NOTE: k has to be a multiple of BK * num_stages. Hard coded this condition to 128 for now.
    # TODO: Need to find a better dispatch strategy.
    var h100_matmul_cond = (
        ctx.device_info is H100 and n % 8 == 0 and a_type is DType.bfloat16
    )
    var amdgpu_matmul_cond = has_amd_gpu_accelerator() and n % 4 == 0
    var multi_gemm_cond = (
        (m > 1 or (has_amd_gpu_accelerator() and transpose_b == False))
        and (n % 128 == 0 or h100_matmul_cond or amdgpu_matmul_cond)
        and k % 32 == 0
        and k >= 128
    )
    # fmt: off
    # Require Static K, N in A, B, C
    alias has_static_NK = b_shape.all_known[2]() \
                      and a_shape.has_value[1]() \
                      and c_shape.has_value[1]()
    # fmt: on

    @parameter
    if ctx.device_info > H100:
        return _matmul_sm100[
            c_type,
            a_type,
            b_type,
            use_tensor_core,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            config=config,
            _trace_description=_trace_description,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    @parameter
    if env_get_bool["MODULE_USE_VENDOR_BLAS", False]():
        return matmul_vendor[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            config=config,
            _trace_description=_trace_description,
        ](c, a, b, ctx)

    @parameter
    if (
        (a_type is DType.float8_e4m3fn)
        and a_type == b_type
        and ctx.device_info is H100
        and transpose_b
        and has_static_NK
    ):

        @parameter
        if env_get_bool["AUTOTUNING_MODE", False]():
            if env_get_bool["H100_SPECIFIC", False]():
                alias NUM_PIPELINE_STAGES = env_get_int[
                    "TUNE_NUM_PIPELINE_STAGES", 4
                ]()
                alias NUM_CONSUMER = env_get_int["TUNE_NUM_CONSUMER", 1]()
                alias WGMMA_N = env_get_int["TUNE_WGMMA_N", 128]()
                alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 1]()
                alias GRID_DIM_X = env_get_int["TUNE_GRID_DIM_X", 1]()
                alias GRID_DIM_Y = H100.sm_count // GRID_DIM_X
                alias BLOCK_TILE_DIM_M = 64 * NUM_CONSUMER

                alias SCHEDULE_TYPE = MatmulSchedule(
                    env_get_int["TUNE_SCHEDULE_TYPE", 1]()
                )

                alias H100_FP8_TUNING_CONFIG = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, WGMMA_N, 32),
                ](
                    block_tile_shape=Index(BLOCK_TILE_DIM_M, WGMMA_N, 128),
                    cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
                    num_pipeline_stages=NUM_PIPELINE_STAGES,
                    num_consumer=NUM_CONSUMER,
                    partitioned_multicast=False,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=H100_FP8_TUNING_CONFIG,
                    grid_shape = Index(128, 1),
                    schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

        alias static_M = c_shape.get[0]()
        alias static_N = c_shape.get[1]()
        alias static_K = a_shape.get[1]()

        # llama-405B-FP8 gemm shapes
        @parameter
        if static_N == 16384 and static_K == 2048:
            if m <= 64:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(64, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(128, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 128:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(H100.sm_count, 1),
                    schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 256:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 512:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(H100.sm_count, 1),
                    schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(8, H100.sm_count // 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

        elif static_N == 2304 and static_K == 16384:
            if m <= 64:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 48, 32),
                ](
                    block_tile_shape=Index(64, 48, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 128:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 48, 32),
                ](
                    block_tile_shape=Index(64, 48, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 256:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 96, 32),
                ](
                    block_tile_shape=Index(64, 96, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 512:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 144, 32),
                ](
                    block_tile_shape=Index(128, 144, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 144, 32),
                ](
                    block_tile_shape=Index(128, 144, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(H100.sm_count, 1),
                    schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 2048:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 144, 32),
                ](
                    block_tile_shape=Index(128, 144, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(16, 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    # grid_shape = Index(16, 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

        elif static_N == 13312 and static_K == 16384:
            if m <= 64:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(64, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(128, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 128:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    # grid_shape = Index(H100.sm_count, 1),
                    # schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 256:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 208, 32),
                ](
                    block_tile_shape=Index(128, 208, 128),
                    cluster_shape=Index(1, 2, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    # schedule = MatmulSchedule.DS_SCHEDULER,
                    # grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 512:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    # schedule = MatmulSchedule.DS_SCHEDULER,
                    # grid_shape = Index(H100.sm_count, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    # grid_shape = Index(H100.sm_count, 1),
                    # schedule = MatmulSchedule.DS_SCHEDULER,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(8, H100.sm_count // 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

        elif static_N == 16384 and static_K == 6656:
            if m <= 64:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(64, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=False,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    schedule = MatmulSchedule.DS_SCHEDULER,
                    grid_shape = Index(128, 1),
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=4,
                    num_consumer=2,
                    partitioned_multicast=True,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(8, H100.sm_count // 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return

        # llama-8B-FP8 gemm shapes
        elif (
            (static_N == 6144 and static_K == 4096)
            or (static_N == 4096 and static_K == 4096)
            or (static_N == 28672 and static_K == 4096)
            or (static_N == 4096 and static_K == 14336)
        ):
            if m <= 128:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(64, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=8,
                    num_consumer=1,
                    partitioned_multicast=True,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            elif m <= 1024:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(1, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=2,
                    partitioned_multicast=True,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
            else:
                alias config = MatmulConfig[
                    a_type,
                    b_type,
                    c_type,
                    transpose_b,
                    mma_shape = Index(64, 128, 32),
                ](
                    block_tile_shape=Index(128, 128, 128),
                    cluster_shape=Index(2, 1, 1),
                    num_pipeline_stages=6,
                    num_consumer=2,
                    partitioned_multicast=True,
                    pdl_level=pdl_level,
                )
                warp_specialize_gemm_with_multicasting[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                    config=config,
                    grid_shape = Index(8, H100.sm_count // 8),
                    schedule = MatmulSchedule.TILE2D,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    m,
                    n,
                    k,
                    ctx,
                )
                return
        else:
            # for gemms with small n and k we fall back the naive kernel
            alias BN = _find_largest_bn_for_sm90_matmul[a_type, static_N]()
            alias BK = 128

            @parameter
            if BN != -1 and static_K % BK == 0:
                # If the number of blocks is less than the number of SMs, it's probably better to not use any persistent kernel
                if ceildiv(m, 64) * ceildiv(static_N, BN) <= H100.sm_count:
                    alias config = MatmulConfig[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        mma_shape = Index(64, BN, 32),
                    ](
                        block_tile_shape=Index(64, BN, BK),
                        cluster_shape=Index(1, 1, 1),
                        num_pipeline_stages=6,
                        num_consumer=1,
                        partitioned_multicast=False,
                    )
                    warp_specialize_gemm_with_multicasting[
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                        config=config,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                        m,
                        n,
                        k,
                        ctx,
                    )
                    return
                elif m <= 1024:
                    alias config = MatmulConfig[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        mma_shape = Index(64, BN, 32),
                    ](
                        block_tile_shape=Index(64, BN, BK),
                        cluster_shape=Index(1, 1, 1),
                        num_pipeline_stages=6,
                        num_consumer=1,
                        partitioned_multicast=False,
                    )
                    warp_specialize_gemm_with_multicasting[
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                        config=config,
                        schedule = MatmulSchedule.DS_SCHEDULER,
                        grid_shape = Index(H100.sm_count, 1),
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                        m,
                        n,
                        k,
                        ctx,
                    )
                    return
                else:
                    alias config = MatmulConfig[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        mma_shape = Index(64, BN, 32),
                    ](
                        block_tile_shape=Index(128, BN, BK),
                        cluster_shape=Index(1, 1, 1),
                        num_pipeline_stages=4,
                        num_consumer=2,
                        partitioned_multicast=False,
                    )
                    warp_specialize_gemm_with_multicasting[
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                        config=config,
                        schedule = MatmulSchedule.DS_SCHEDULER,
                        grid_shape = Index(H100.sm_count, 1),
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                        m,
                        n,
                        k,
                        ctx,
                    )
                    return

    @parameter
    if (
        matmul_supported_format
        and has_accelerator()
        and use_tensor_core
        and has_static_NK
    ):
        if multi_gemm_cond:
            alias kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

            @parameter
            if has_amd_gpu_accelerator():
                alias static_N = c_shape.get[1]()
                alias static_K = a_shape.get[1]()

                @always_inline
                @parameter
                fn scheduler_hint_helper[
                    block_m: Int, block_n: Int
                ]() -> IndexList[3]:
                    alias table = List[AMDSchedulerTuning](
                        # Entered by hand
                        AMDSchedulerTuning(Index(32, 32), Index(3, 7, 1)),
                        AMDSchedulerTuning(Index(32, 64), Index(1, 4, 2)),
                        AMDSchedulerTuning(Index(32, 96), Index(5, 7, 1)),
                        AMDSchedulerTuning(Index(32, 128), Index(1, 1, 2)),
                        AMDSchedulerTuning(Index(32, 160), Index(3, 5, 5)),
                        AMDSchedulerTuning(Index(32, 192), Index(5, 6, 5)),
                        AMDSchedulerTuning(Index(32, 224), Index(2, 3, 5)),
                        AMDSchedulerTuning(Index(32, 256), Index(2, 1, 1)),
                        AMDSchedulerTuning(Index(64, 32), Index(1, 9, 1)),
                        AMDSchedulerTuning(Index(64, 64), Index(3, 1, 1)),
                        AMDSchedulerTuning(Index(64, 96), Index(4, 1, 1)),
                        AMDSchedulerTuning(Index(64, 128), Index(3, 1, 2)),
                        AMDSchedulerTuning(Index(64, 160), Index(1, 7, 1)),
                        AMDSchedulerTuning(Index(64, 192), Index(1, 1, 2)),
                        AMDSchedulerTuning(Index(64, 224), Index(2, 1, 3)),
                        AMDSchedulerTuning(Index(64, 256), Index(2, 5, 2)),
                        AMDSchedulerTuning(Index(96, 32), Index(3, 8, 1)),
                        AMDSchedulerTuning(Index(96, 64), Index(1, 3, 2)),
                        AMDSchedulerTuning(Index(96, 96), Index(2, 4, 1)),
                        AMDSchedulerTuning(Index(96, 128), Index(1, 2, 2)),
                        AMDSchedulerTuning(Index(96, 160), Index(4, 2, 1)),
                        AMDSchedulerTuning(Index(96, 192), Index(1, 4, 1)),
                        AMDSchedulerTuning(Index(96, 224), Index(4, 2, 2)),
                        AMDSchedulerTuning(Index(96, 256), Index(4, 1, 2)),
                        AMDSchedulerTuning(Index(128, 32), Index(1, 5, 1)),
                        AMDSchedulerTuning(Index(128, 64), Index(1, 2, 1)),
                        AMDSchedulerTuning(Index(128, 96), Index(1, 4, 1)),
                        AMDSchedulerTuning(Index(128, 128), Index(3, 1, 2)),
                        AMDSchedulerTuning(Index(128, 160), Index(3, 3, 2)),
                        AMDSchedulerTuning(Index(128, 192), Index(2, 4, 2)),
                        AMDSchedulerTuning(Index(128, 224), Index(5, 1, 2)),
                        AMDSchedulerTuning(Index(128, 256), Index(2, 5, 2)),
                        # Auto generated with kprofile
                        AMDSchedulerTuning(Index(160, 32), Index(2, 2, 1)),
                        AMDSchedulerTuning(Index(160, 64), Index(3, 1, 1)),
                        AMDSchedulerTuning(Index(160, 96), Index(3, 1, 2)),
                        AMDSchedulerTuning(Index(160, 128), Index(4, 2, 1)),
                        AMDSchedulerTuning(Index(160, 160), Index(4, 4, 1)),
                        AMDSchedulerTuning(Index(160, 192), Index(6, 1, 2)),
                        AMDSchedulerTuning(Index(160, 224), Index(7, 1, 2)),
                        AMDSchedulerTuning(Index(160, 256), Index(8, 1, 2)),
                        AMDSchedulerTuning(Index(192, 32), Index(2, 2, 1)),
                        AMDSchedulerTuning(Index(192, 64), Index(1, 2, 2)),
                        AMDSchedulerTuning(Index(192, 96), Index(2, 3, 2)),
                        AMDSchedulerTuning(Index(192, 128), Index(3, 3, 2)),
                        AMDSchedulerTuning(Index(192, 160), Index(8, 1, 1)),
                        AMDSchedulerTuning(Index(192, 192), Index(2, 5, 2)),
                        AMDSchedulerTuning(Index(192, 224), Index(5, 4, 2)),
                        AMDSchedulerTuning(Index(192, 256), Index(7, 3, 2)),
                        AMDSchedulerTuning(Index(224, 32), Index(3, 1, 1)),
                        AMDSchedulerTuning(Index(224, 64), Index(1, 1, 2)),
                        AMDSchedulerTuning(Index(224, 96), Index(3, 1, 2)),
                        AMDSchedulerTuning(Index(224, 128), Index(6, 1, 2)),
                        AMDSchedulerTuning(Index(224, 160), Index(3, 5, 2)),
                        AMDSchedulerTuning(Index(224, 192), Index(6, 2, 2)),
                        AMDSchedulerTuning(Index(224, 224), Index(6, 4, 2)),
                        AMDSchedulerTuning(Index(224, 256), Index(4, 7, 2)),
                        AMDSchedulerTuning(Index(256, 32), Index(1, 2, 2)),
                        AMDSchedulerTuning(Index(256, 64), Index(2, 1, 2)),
                        AMDSchedulerTuning(Index(256, 96), Index(4, 1, 2)),
                        AMDSchedulerTuning(Index(256, 128), Index(6, 2, 1)),
                        AMDSchedulerTuning(Index(256, 160), Index(3, 6, 2)),
                        AMDSchedulerTuning(Index(256, 192), Index(6, 3, 2)),
                        AMDSchedulerTuning(Index(256, 224), Index(4, 6, 2)),
                        AMDSchedulerTuning(Index(256, 256), Index(6, 6, 2)),
                    )

                    @parameter
                    if (
                        env_get_bool["AUTOTUNING_MODE", False]()
                        and env_get_bool["AMD_SCHEDULER_TUNING", False]()
                    ):
                        return Index(
                            env_get_int["TUNE_SCHED_X", 2](),
                            env_get_int["TUNE_SCHED_Y", 2](),
                            env_get_int["TUNE_SCHED_Z", 2](),
                        )

                    @parameter
                    for i in range(len(table)):
                        if table[i].block_shape == Index(block_m, block_n):
                            return table[i].tuning_values
                    return Index(2, 2, 2)

                @always_inline
                @parameter
                fn kernel_helper[
                    block_m: Int,
                    block_n: Int,
                    *,
                    num_k_partitions: Int = 1,
                    num_pipeline_stages: Int = 1,
                ]() raises:
                    alias config = MatmulConfig[
                        a_type, b_type, c_type, transpose_b
                    ](
                        block_tile_shape=Index(
                            block_m, block_n, _bk_base[a_type, True]()
                        ),
                        warp_tile_shape=Index(
                            block_m // 2, block_n // 2, _bk_base[a_type, True]()
                        ),
                        num_pipeline_stages=num_pipeline_stages,
                        scheduler_hint=scheduler_hint_helper[
                            block_m, block_n
                        ](),
                        num_k_partitions=num_k_partitions,
                        pdl_level=pdl_level,
                    )
                    multistage_gemm[
                        transpose_b=transpose_b,
                        config=config,
                        elementwise_lambda_fn=elementwise_lambda_wrapper,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                        config,
                        ctx,
                    )

                @parameter
                if not transpose_b:
                    return kernel_helper[128, 128, num_pipeline_stages=2]()
                elif env_get_bool["AUTOTUNING_MODE", False]():
                    alias block_m = env_get_int["TUNE_BM", 128]()
                    alias block_n = env_get_int["TUNE_BN", 128]()
                    alias num_k_partitions = env_get_int[
                        "TUNE_NUM_K_PARTITIONS", 1
                    ]()
                    return kernel_helper[
                        block_m, block_n, num_k_partitions=num_k_partitions
                    ]()

                # mistral-small-24b auto-tuned shapes
                @parameter
                if static_N == 5120 and static_K == 4096:
                    if m >= 8192:
                        return kernel_helper[192, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 256]()
                    elif m >= 3500:
                        return kernel_helper[256, 256]()
                    elif m >= 512:
                        return kernel_helper[96, 64]()
                    elif m >= 500:
                        return kernel_helper[128, 160, num_k_partitions=2]()
                    elif m >= 256:
                        return kernel_helper[128, 160, num_k_partitions=4]()
                elif static_N == 6144 and static_K == 5120:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[192, 256]()
                    elif m >= 3500:
                        return kernel_helper[192, 192]()
                    elif m >= 512:
                        return kernel_helper[192, 128, num_k_partitions=2]()
                    elif m >= 500:
                        return kernel_helper[96, 128]()
                    elif m >= 256:
                        return kernel_helper[128, 192, num_k_partitions=4]()
                elif static_N == 65536 and static_K == 5120:
                    if m > 384:
                        return kernel_helper[256, 256]()
                    elif m > 256:
                        return kernel_helper[192, 256]()
                    elif m > 192:
                        return kernel_helper[256, 256]()
                    elif m > 128:
                        return kernel_helper[192, 256]()
                    elif m > 64:
                        return kernel_helper[128, 256]()
                elif static_N == 5120 and static_K == 32768:
                    if m >= 8192:
                        return kernel_helper[192, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 256]()
                    elif m >= 3500:
                        return kernel_helper[256, 256]()
                    elif m >= 512:
                        return kernel_helper[256, 160, num_k_partitions=4]()
                    elif m >= 500:
                        return kernel_helper[192, 224, num_k_partitions=4]()
                    elif m >= 256:
                        return kernel_helper[128, 96, num_k_partitions=8]()

                # gemma-3-12b auto-tuned shapes
                @parameter
                if static_N == 3840 and static_K == 4096:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[192, 256]()
                    elif m >= 3500:
                        return kernel_helper[256, 192]()
                    elif m >= 512:
                        return kernel_helper[96, 160, num_k_partitions=2]()
                    elif m >= 500:
                        return kernel_helper[96, 160, num_k_partitions=2]()
                    elif m >= 256:
                        return kernel_helper[128, 128, num_k_partitions=4]()
                elif static_N == 3840 and static_K == 15360:
                    if m >= 8192:
                        return kernel_helper[256, 224]()
                    elif m >= 7000:
                        return kernel_helper[224, 224]()
                    elif m >= 3500:
                        return kernel_helper[224, 224]()
                    elif m >= 512:
                        return kernel_helper[96, 160, num_k_partitions=4]()
                    elif m >= 500:
                        return kernel_helper[96, 160, num_k_partitions=4]()
                    elif m >= 256:
                        return kernel_helper[128, 224, num_k_partitions=8]()
                elif static_N == 8192 and static_K == 3840:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 256]()
                    elif m >= 3500:
                        return kernel_helper[192, 256]()
                    elif m >= 512:
                        return kernel_helper[128, 128, num_k_partitions=2]()
                    elif m >= 500:
                        return kernel_helper[128, 128, num_k_partitions=2]()
                    elif m >= 256:
                        return kernel_helper[64, 64]()
                elif static_N == 30720 and static_K == 3840:
                    if m >= 8192:
                        return kernel_helper[256, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 256]()
                    elif m >= 3500:
                        return kernel_helper[256, 256]()
                    elif m >= 512:
                        return kernel_helper[256, 256]()
                    elif m >= 500:
                        return kernel_helper[256, 224]()
                    elif m >= 256:
                        return kernel_helper[128, 224]()
                elif static_N == 262208 and static_K == 3840:
                    return kernel_helper[256, 256]()

                # llama3 auto-tuned shapes
                @parameter
                if static_N == 4096 and static_K == 4096:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 224]()
                    elif m >= 3500:
                        return kernel_helper[192, 256]()
                    elif m >= 512:
                        return kernel_helper[64, 64]()
                    elif m >= 500:
                        return kernel_helper[64, 64]()
                    elif m >= 256:
                        return kernel_helper[128, 128, num_k_partitions=4]()
                elif static_N == 4096 and static_K == 14336:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[192, 256]()
                    elif m >= 3500:
                        return kernel_helper[192, 256]()
                    elif m >= 512:
                        return kernel_helper[256, 224, num_k_partitions=8]()
                    elif m >= 500:
                        return kernel_helper[256, 224, num_k_partitions=8]()
                    elif m >= 256:
                        return kernel_helper[128, 224, num_k_partitions=8]()
                elif static_N == 6144 and static_K == 4096:
                    if m >= 8192:
                        return kernel_helper[224, 256]()
                    elif m >= 7000:
                        return kernel_helper[192, 256]()
                    elif m >= 3500:
                        return kernel_helper[192, 192]()
                    elif m >= 512:
                        return kernel_helper[192, 128, num_k_partitions=2]()
                    elif m >= 500:
                        return kernel_helper[96, 128]()
                    elif m >= 256:
                        return kernel_helper[128, 192, num_k_partitions=4]()
                elif static_N == 28672 and static_K == 4096:
                    if m >= 8192:
                        return kernel_helper[256, 256]()
                    elif m >= 7000:
                        return kernel_helper[256, 256]()
                    elif m >= 3500:
                        return kernel_helper[224, 256]()
                    elif m >= 512:
                        return kernel_helper[256, 192]()
                    elif m >= 500:
                        return kernel_helper[256, 192]()
                    elif m >= 256:
                        return kernel_helper[128, 96]()

                # Default tune based on llama3
                @parameter
                if static_N >= 28672 and static_K >= 2048:
                    if m >= 1024:
                        return kernel_helper[224, 256]()
                    elif m >= 128:
                        return kernel_helper[128, 128]()
                    else:
                        return kernel_helper[64, 64]()
                elif static_N >= 2048 and static_K >= 2048:
                    if m >= 4096:
                        return kernel_helper[224, 256]()
                    elif m >= 1024:
                        return kernel_helper[128, 128]()
                    elif m >= 512:
                        return kernel_helper[128, 128, num_k_partitions=2]()
                    elif static_K == 14336:
                        if m >= 128:
                            return kernel_helper[64, 64, num_k_partitions=4]()
                        elif m >= 64:
                            return kernel_helper[64, 64, num_k_partitions=8]()
                        else:
                            return kernel_helper[32, 64, num_k_partitions=4]()
                    elif m >= 64:
                        return kernel_helper[64, 64]()
                    else:
                        return kernel_helper[32, 64, num_k_partitions=4]()
                return kernel_helper[128, 128]()

            @parameter
            if env_get_bool["AUTOTUNING_MODE", False]():
                if env_get_bool["H100_SPECIFIC", False]():
                    # CLUSTER_DIM_X = 2^m for m in range[0-3]
                    alias CLUSTER_DIM_X = env_get_int["TUNE_CLUSTER_DIM_X", 1]()

                    # GRID_DIM_X = 2^n for n in range[0-7]
                    alias GRID_DIM_X = env_get_int["TUNE_GRID_DIM_X", 1]()
                    alias GRID_DIM_Y = H100.sm_count // GRID_DIM_X

                    alias H100_TUNING_CONFIG = MatmulConfig[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        mma_shape = Index(64, 256, 16),
                    ](
                        block_tile_shape=Index(128, 256, 64),
                        cluster_shape=Index(CLUSTER_DIM_X, 1, 1),
                        num_pipeline_stages=4,
                        num_consumer=2,
                        partitioned_multicast=False,
                        pdl_level=pdl_level,
                    )
                    warp_specialize_gemm_with_multicasting[
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                        config=H100_TUNING_CONFIG,
                        grid_shape = Index(GRID_DIM_X, GRID_DIM_Y),
                        schedule = MatmulSchedule.TILE2D,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                        m,
                        n,
                        k,
                        ctx,
                    )
                    return

                else:
                    multistage_gemm[
                        transpose_b=transpose_b,
                        config = kernels.tuning_config,
                        elementwise_lambda_fn=elementwise_lambda_wrapper,
                    ](
                        rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                        rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                        rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                and (a_type.is_half_float() or a_type is DType.float32)
                and ctx.device_info is H100
                and transpose_b
                and not use_A100_kernels_on_H100
            ):
                alias static_N = c_shape.get[1]()
                alias static_K = a_shape.get[1]()
                alias a_is_bfloat16_or_float32 = a_type in (
                    DType.bfloat16,
                    DType.float32,
                )
                alias size_factor = 2 if a_type is DType.float32 else 1
                alias mma_k = 16 // size_factor
                alias BK = 64 // size_factor

                # GTC matmul configs
                @parameter
                if (
                    a_is_bfloat16_or_float32
                    and static_N == 2560
                    and static_K == 8192
                ):
                    if m == 512:
                        alias M512_N2560_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 80 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 80 // size_factor, BK),
                            cluster_shape=Index(1, 2, 1),
                            num_pipeline_stages=8,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M512_N2560_K8192_config,
                            # grid_shape = Index(32, 4),
                            # schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                    elif m == 8192:
                        alias M8192_N2560_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(1, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M8192_N2560_K8192_config,
                            grid_shape = Index(10, H100.sm_count // 10),
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M4096_N2560_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if (
                    a_is_bfloat16_or_float32
                    and static_N == 8192
                    and static_K == 2048
                ):
                    if m == 8192:
                        alias M8192_N8192_K2048_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M8192_N8192_K2048_config,
                            grid_shape = Index(4, H100.sm_count // 4),
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M4096_N8192_K2048_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if (
                    a_is_bfloat16_or_float32
                    and static_N == 14336
                    and static_K == 8192
                ):
                    if m == 8192:
                        alias M8192_N14336_K8192_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M8192_N14336_K8192_config,
                            grid_shape = Index(8, H100.sm_count // 8),
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M4096_N14336_K8192_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                @parameter
                if (
                    a_is_bfloat16_or_float32
                    and static_N == 8192
                    and static_K == 7168
                ):
                    if m == 8192:
                        alias M8192_N8192_K7168_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M8192_N8192_K7168_config,
                            grid_shape = Index(8, H100.sm_count // 8),
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
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
                            mma_shape = Index(64, 256 // size_factor, mma_k),
                        ](
                            block_tile_shape=Index(128, 256 // size_factor, BK),
                            cluster_shape=Index(2, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=M4096_N8192_K7168_config,
                            schedule = MatmulSchedule.TILE2D,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

                alias BN = _find_largest_bn_for_sm90_matmul[
                    a_type, static_N
                ]() // size_factor

                # `audio_decoder/test_residual_fsq.py::test_fsq` test fails if
                # we enable float32 here.
                @parameter
                if a_type is DType.bfloat16 and BN != -1 and static_K % BK == 0:
                    if m <= 128:
                        alias default_bf16_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, BN, mma_k),
                        ](
                            block_tile_shape=Index(64, BN, BK),
                            cluster_shape=Index(1, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=1,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=default_bf16_config,
                            schedule = MatmulSchedule.NONE,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return
                    else:
                        alias default_bf16_config = MatmulConfig[
                            a_type,
                            b_type,
                            c_type,
                            transpose_b,
                            mma_shape = Index(64, BN, mma_k),
                        ](
                            block_tile_shape=Index(128, BN, BK),
                            cluster_shape=Index(1, 1, 1),
                            num_pipeline_stages=4,
                            num_consumer=2,
                            partitioned_multicast=False,
                            pdl_level=pdl_level,
                        )
                        warp_specialize_gemm_with_multicasting[
                            transpose_b=transpose_b,
                            elementwise_lambda_fn=elementwise_lambda_fn,
                            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                            config=default_bf16_config,
                            schedule = MatmulSchedule.NONE,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            m,
                            n,
                            k,
                            ctx,
                        )
                        return

            alias use_A100_kernels = ctx.device_info is A100 or (
                ctx.device_info is H100 and use_A100_kernels_on_H100 != 0
            )

            @parameter
            if (
                a_type == b_type
                and a_type.is_half_float()
                and use_A100_kernels
                and transpose_b
            ):
                alias static_K = a_shape.get[1]()
                alias static_N = c_shape.get[1]()
                alias Ms = InlineArray[Int32, 10](
                    16, 32, 64, 128, 256, 512, 768, 1024, 2048, 4096
                )
                try:

                    @parameter
                    for k in range(0, 10):
                        alias M = Ms[k]
                        if M <= m:
                            alias key = String(M, "_", static_N, "_", static_K)
                            alias curr_config = create_matmul_configs_ampere[
                                key, a_type, b_type, c_type, transpose_b
                            ]()
                            if curr_config.num_pipeline_stages == 0:
                                raise "no match for the triple"
                            else:
                                multistage_gemm[
                                    c_type=c_type,
                                    c_shape=c_shape,
                                    a_type=a_type,
                                    a_shape=a_shape,
                                    b_type=b_type,
                                    b_shape=b_shape,
                                    transpose_b=transpose_b,
                                    config=curr_config,
                                    elementwise_lambda_fn=elementwise_lambda_fn,
                                ](
                                    rebind[
                                        NDBuffer[c_type, 2, c.origin, c_shape]
                                    ](c),
                                    rebind[
                                        NDBuffer[a_type, 2, a.origin, a_shape]
                                    ](a),
                                    rebind[
                                        NDBuffer[b_type, 2, b.origin, b_shape]
                                    ](b),
                                    curr_config,
                                    ctx,
                                )
                                return
                    raise "no match for the triple"
                except:
                    var best_config = select_config[
                        a_type, b_type, c_type, transpose_b
                    ](m, n, k, ctx)

                    if best_config == kernels.ampere_256x64_4:
                        alias config = kernels.ampere_256x64_4
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config=config,
                            elementwise_lambda_fn=elementwise_lambda_wrapper,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            best_config,
                            ctx,
                        )

                    elif best_config == kernels.ampere_256x128_3:
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config = kernels.ampere_256x128_3,
                            elementwise_lambda_fn=elementwise_lambda_wrapper,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            best_config,
                            ctx,
                        )

                    else:  # Default kernel 128x128_4
                        multistage_gemm[
                            transpose_b=transpose_b,
                            config = kernels.ampere_128x128_4,
                            elementwise_lambda_fn=elementwise_lambda_wrapper,
                        ](
                            rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                            rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                            rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                            best_config,
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
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    best_config,
                    ctx,
                )

            elif best_config == kernels.ampere_256x128_3:
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.ampere_256x128_3,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    best_config,
                    ctx,
                )

            else:  # Default kernel 128x128_4
                multistage_gemm[
                    transpose_b=transpose_b,
                    config = kernels.ampere_128x128_4,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    best_config,
                    ctx,
                )

            return

    @parameter
    if not a_type.is_float8():
        if n == 1 or m == 1:
            gemv_gpu[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ](c, a, b, ctx)
            return

    # compile time check to pass bro's req of only having support FP32, FP16 and BF16 for cublas wrapper
    @parameter
    if (
        (
            a_type is DType.float32
            or a_type is DType.bfloat16
            or a_type is DType.float16
        )
        and (
            b_type is DType.float32
            or b_type is DType.bfloat16
            or b_type is DType.float16
        )
        and (
            c_type is DType.float32
            or c_type is DType.bfloat16
            or c_type is DType.float16
        )
    ):
        try:
            return matmul_vendor[
                use_tensor_core=use_tensor_core,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                config=config,
                _trace_description=_trace_description,
            ](c, a, b, ctx)
        except:
            alias BLOCK_DIM = 16
            ctx.enqueue_function[
                matmul_kernel_naive[
                    c_type,
                    a_type,
                    b_type,
                    BLOCK_DIM,
                    transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
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
            return
    else:
        # For unsupported types like FP8, directly use the naive implementation
        alias BLOCK_DIM = 16
        ctx.enqueue_function[
            matmul_kernel_naive[
                c_type,
                a_type,
                b_type,
                BLOCK_DIM,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
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
        return


@always_inline
fn split_k_reduce[
    c_type: DType,
    work_space_type: DType,
    c_shape: DimList,
    work_space_shape: DimList,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, c_shape],
    work_space: NDBuffer[work_space_type, 3, _, work_space_shape],
    ctx: DeviceContext,
) raises:
    alias simd_width = simdwidthof[c_type, target = get_gpu_target()]()
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
    c: NDBuffer[mut=True, c_type, 2, _, c_shape],
    a: NDBuffer[a_type, 2, _, a_shape],
    b: NDBuffer[b_type, 2, _, b_shape],
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
                c_type is DType.bfloat16,
                "serial reduction is unsupported for this config",
            ]()
            alias work_space_type = config.split_k_reduction_type

            # For the serial reduction we don't use workspace
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
                locks_data,
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
                work_space_data._unsafe_ptr(),
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

            @parameter
            if has_amd_gpu_accelerator():
                ctx.enqueue_function[gemm_kernel_type](
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    work_space,
                    runtime_config.num_k_partitions,
                    UnsafePointer[Int32](),
                    grid_dim=runtime_config.grid_dim(M, N),
                    block_dim=runtime_config.block_dim(),
                )
            else:
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
            tensor_c.layout_int_type,
            tensor_a.layout_int_type,
            tensor_b.layout_int_type,
            tensor_c.linear_idx_type,
            tensor_a.linear_idx_type,
            tensor_b.linear_idx_type,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
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
            tensor_c.layout_int_type,
            tensor_a.layout_int_type,
            tensor_b.layout_int_type,
            tensor_c.linear_idx_type,
            tensor_a.linear_idx_type,
            tensor_b.linear_idx_type,
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
