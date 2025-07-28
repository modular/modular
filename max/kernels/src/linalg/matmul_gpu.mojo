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
    env_get_bool,
    env_get_int,
    has_accelerator,
    has_amd_gpu_accelerator,
    llvm_intrinsic,
    simdwidthof,
)
from sys import sizeof
from algorithm.functional import elementwise, tile_and_unswitch
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    barrier,
    block_dim,
    global_idx,
    thread_idx,
)
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext, FuncAttribute
from gpu.host import get_gpu_target
from gpu.host.info import A100, H100, B200
from gpu.memory import AddressSpace
from layout._ndbuffer_stub import (
    from_ndbuffer_row_major,
)
from layout.layout import *
from logger import Logger
from memory import bitcast, stack_allocation

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type

from .matmul_amd import gemm_kernel_amd
from ._multistage_gemm_gpu import (
    multistage_gemm_kernel,
    multistage_gemm_split_k_kernel,
)
from .dispatch_table_a100_gpu import create_matmul_configs_ampere
from .gemv import gemv_gpu
from .matmul_vendor import matmul as matmul_vendor
from .matmul_dispatch_sm90 import matmul_dispatch_sm90
from .matmul_sm100 import matmul_sm100_fallback
from .utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    select_config,
)


@always_inline
fn __nvvm_ldg_f4[
    dtype: DType
](x: UnsafePointer[Scalar[dtype]]) -> SIMD[dtype, 4]:
    # Load a register variable from global state space via non-coherent cache.

    alias alignment = Int32(alignof[SIMD[dtype, 4]]())

    @parameter
    if dtype is DType.float32:
        return bitcast[dtype, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4f32.p0v4f32", SIMD[DType.float32, 4]
            ](x.bitcast[Float32](), alignment)
        )
    elif dtype is DType.bfloat16:
        return bitcast[dtype, 4](
            llvm_intrinsic[
                "llvm.nvvm.ldg.global.f.v4bf16.p0v4bf16",
                SIMD[DType.bfloat16, 4],
            ](x.bitcast[BFloat16](), alignment)
        )
    elif dtype is DType.float16:
        return bitcast[dtype, 4](
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
            a_val = a[Int(row), Int(offset + localCol)] if (
                row < m and offset + localCol < k
            ) else 0.0
        else:
            a_val = a[Int(row), Int(offset + localCol)] if row < m else 0.0
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Scalar[b_type]

        @parameter
        if not full_tile:
            b_val = b[Int(offset + localRow), Int(col)] if (
                col < n and offset + localRow < k
            ) else 0.0
        else:
            b_val = b[Int(offset + localRow), Int(col)] if col < n else 0.0
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
    var x = Int(global_idx.x)
    var y = Int(global_idx.y)

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

    var logger = Logger()
    logger.info("------ Dispatching to SM100 (B200+) ------")

    try:
        # On B200 our gemv matmul is faster than cublas for skinny bfloat16 matmuls
        @parameter
        if a_type is DType.bfloat16:
            if n == 1 or m == 1:
                return gemv_gpu[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, ctx)

        logger.info("Executing vendor BLAS (cuBLAS)")
        return matmul_vendor[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            config=config,
            _trace_description=_trace_description,
        ](c, a, b, ctx)

    except:
        # fallback to multistage/naive gemms if the cublas failed. This is a workaround for now for KERN-1812
        logger.warning("Vendor BLAS failed")

        @parameter
        if not a_type.is_float8() and K * sizeof[a_type]() >= 8 * 16:
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
            logger.info(
                "Executing: Naive MATMUL kernel (BLOCK_DIM=", BLOCK_DIM, ")"
            )
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

    var logger = Logger()
    logger.info("---- MATMUL GPU execution started ----")
    logger.info("MxNxK: ", m, "x", n, "x", k)
    logger.info("Data types: A=", a_type, " B=", b_type, " C=", c_type)
    logger.info("Device: ", ctx.name())
    logger.info(
        "Transpose B: ", transpose_b, " Use Tensor Core: ", use_tensor_core
    )

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
        _dtype: DType, _width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_dtype, _width]):
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
        ctx.default_device_info is H100
        and n % 8 == 0
        and a_type is DType.bfloat16
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

    logger.info("Static shapes available: N=", b_shape.has_value[1](), " K=", a_shape.has_value[1]())
    # fmt: on

    @parameter
    if env_get_bool["MODULE_USE_VENDOR_BLAS", False]():
        logger.info("Executing: Vendor BLAS")
        return matmul_vendor[
            use_tensor_core=use_tensor_core,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            config=config,
            _trace_description=_trace_description,
        ](c, a, b, ctx)

    alias use_experimental_kernels = Bool(
        env_get_int["USE_EXPERIMENTAL_KERNELS", 0]()
    )

    alias bf16_or_fp16 = (DType.bfloat16, DType.float16)
    alias bf16_or_fp16_fp32 = (DType.bfloat16, DType.float16, DType.float32)

    @parameter
    if (
        ctx.default_device_info is B200
        and use_experimental_kernels
        and transpose_b
        and (
            a_type in bf16_or_fp16
            and b_type in bf16_or_fp16
            and c_type in bf16_or_fp16_fp32
        )
    ):
        var a_layout_tensor = from_ndbuffer_row_major(a)
        var b_layout_tensor = from_ndbuffer_row_major(b)
        var c_layout_tensor = from_ndbuffer_row_major(c)
        alias umma_shape = Index(64, 128, 16)
        alias BK = 64
        alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)
        return matmul_sm100_fallback[
            transpose_b=transpose_b,
            umma_shape=umma_shape,
            block_tile_shape=block_tile_shape,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](c_layout_tensor, a_layout_tensor, b_layout_tensor, ctx)

    @parameter
    if ctx.default_device_info > H100:
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
    if ctx.default_device_info is H100:
        var status = matmul_dispatch_sm90[
            c_type,
            a_type,
            b_type,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

        if status:
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

            alias use_A100_kernels = ctx.default_device_info is A100

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
        # to disable vendor fallback, run export MODULAR_DISABLE_VENDOR_FALLBACK=1 in the environment
        and not env_get_bool["MODULAR_DISABLE_VENDOR_FALLBACK", False]()
    ):
        logger.info("Executing: vendor BLAS fallback")
        try:
            return matmul_vendor[
                use_tensor_core=use_tensor_core,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                config=config,
                _trace_description=_trace_description,
            ](c, a, b, ctx)
        except:
            logger.warning("Vendor BLAS failed")
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
        # For unsupported dtypes like FP8, directly use the naive implementation
        logger.info("Unsupported dtypes or vendor disabled")
        alias BLOCK_DIM = 16
        logger.info(
            "Executing: Naive MATMUL kernel (BLOCK_DIM=", BLOCK_DIM, ")"
        )
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

    var logger = Logger()
    logger.info("------ Dispatching to Multistage GEMM ------")
    logger.info(String(config))
    logger.info("K partitions: ", runtime_config.num_k_partitions)
    logger.info("Serial reduction: ", serial_reduction)

    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b = from_ndbuffer_row_major(b)

    if runtime_config.num_k_partitions > 1:

        @parameter
        if serial_reduction:
            logger.info("Executing: split-K with serial reduction (lock-based)")
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
            logger.info(
                "Executing: split-K with parallel reduction (workspace-based)"
            )
            alias work_space_type = config.split_k_reduction_type
            var work_space_data = ctx.enqueue_create_buffer[work_space_type](
                Int(runtime_config.num_k_partitions * M * N)
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
        logger.info("Executing: AMD standard GEMM (no split-K)")
        alias gemm_kernel_type = gemm_kernel_amd[
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
        logger.info("Executing: standard GEMM (no split-K)")
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
