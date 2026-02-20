# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from math import align_down, ceildiv
from sys import (
    align_of,
    env_get_bool,
    env_get_int,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)
from sys.info import _accelerator_arch

from algorithm.functional import elementwise, tile_and_unswitch
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import barrier, block_dim, global_idx, thread_idx
from gpu.primitives.grid_controls import PDLLevel
from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from gpu.host.info import A100, B200, H100, MI355X, GPUInfo
from layout import LayoutTensor, RuntimeLayout
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout import *
from layout.tensor_core import get_mma_shape
from logger import Logger
from memory import bitcast, stack_allocation
from utils import Index, IndexList
from utils.numerics import get_accum_type

from ...gemv import gemv_gpu
from ...utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ...utils_gpu import (
    MatmulConfig,
    MatmulKernels,
    _bk_base,
    select_config,
    _vendor_blas_fallback_disabled,
)
from ..vendor.matmul import matmul as matmul_vendor
from ._multistage_gemm_gpu import (
    multistage_gemm_kernel,
    multistage_gemm_split_k_kernel,
)
from .amd import gemm_kernel_amd
from .amd_rdna import gemm_kernel_rdna
from .sm80.dispatch import create_matmul_configs_ampere
from .sm90.dispatch import matmul_dispatch_sm90
from .sm100_structured.default.dispatch import matmul_dispatch_sm100
from .sm100_structured.default.matmul import matmul_sm100_fallback

comptime logger = Logger()


fn matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c_ptr: UnsafePointer[mut=True, Scalar[c_type], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[a_type], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
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

        comptime if not full_tile:
            a_val = a[Int(row), offset + Int(localCol)] if (
                row < UInt(m) and offset + Int(localCol) < k
            ) else 0.0
        else:
            a_val = (
                a[Int(row), offset + Int(localCol)] if row < UInt(m) else 0.0
            )
        a_shared[localRow * UInt(tile_size) + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Scalar[b_type]

        comptime if not full_tile:
            b_val = b[offset + Int(localRow), Int(col)] if (
                col < UInt(n) and offset + Int(localRow) < k
            ) else 0.0
        else:
            b_val = (
                b[offset + Int(localRow), Int(col)] if col < UInt(n) else 0.0
            )
        b_shared[localRow * UInt(tile_size) + localCol] = b_val

        barrier()

        for kk in range(tile_size):
            result += (
                a_shared[localRow * UInt(tile_size) + UInt(kk)].cast[s_type]()
                * b_shared[kk * tile_size + Int(localCol)].cast[s_type]()
            )

        barrier()

    tile_and_unswitch[update_tile](
        0, k, VariadicList[Int](tile_size, K_remainder)
    )

    if row < UInt(m) and col < UInt(n):
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(row, col), result.cast[c_type]()
            )
        else:
            c[Index(row, col)] = result.cast[c_type]()


fn matmul_kernel_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    var x = Int(global_idx.x)
    var y = Int(global_idx.y)

    if x >= m or y >= n:
        return

    var accum = Scalar[s_type]()

    comptime if transpose_b:
        for i in range(k):
            var a_val = a[x, i]
            accum += rebind[Scalar[s_type]](a[x, i].cast[s_type]()) * rebind[
                Scalar[s_type]
            ](b[y, i].cast[s_type]())

    else:
        for i in range(k):
            accum += rebind[Scalar[s_type]](a[x, i].cast[s_type]()) * rebind[
                Scalar[s_type]
            ](b[i, y].cast[s_type]())

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[x, y] = accum.cast[c_type]()


fn _amdgpu_get_mma_shape[dtype: DType, transpose_b: Bool]() -> IndexList[3]:
    comptime if transpose_b and _accelerator_arch() == "amdgpu:gfx950":
        comptime if dtype.is_half_float():
            return Index(16, 16, 32)

    return get_mma_shape[dtype, DType.float32]()


fn _amdgpu_matmul_config_from_block_shape[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    K: Int,
    pdl_level: PDLLevel = PDLLevel(),
](block_shape: IndexList[2]) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    comptime max_num_warps: UInt = 4

    var block_m = block_shape[0]
    var block_n = block_shape[1]
    var block_k = _bk_base[a_type, True]()
    var num_warps: UInt = 1
    var num_warp_k_partitions: UInt = 1

    # TODO(KERN-2432): Merge these configurations into the below logic.
    if block_m == 16 and a_type.is_float8() and transpose_b:
        if block_n == 32:
            return MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(16, 32, 256),
                warp_tile_shape=Index(16, 32, 256),
                mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
                num_pipeline_stages=1,
                num_warp_k_partitions=4,
                pdl_level=pdl_level,
            )
        if block_n == 64:
            return MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(16, 64, 1024),
                warp_tile_shape=Index(16, 16, 1024),
                mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
                num_pipeline_stages=1,
                num_warp_k_partitions=1,
                pdl_level=pdl_level,
            )

    if block_m <= 32 and block_n <= 32:
        # Attempt to increase the number of warp_k partitions to improve processor
        # utilization. A single warp needs to read two block_k buffers, so double
        # that in order to expand the number of warp_k partitions.
        var test_k = 2 * (block_k * 2)
        while num_warps < max_num_warps and (K % test_k) == 0:
            num_warp_k_partitions *= 2
            num_warps *= 2
            test_k *= 2
    else:
        # Improve shared memory utilization by expanding block_k, but only if K is
        # a multiple of that expanded block_k size.
        if (K % (block_k * 2)) == 0:
            var smem_a = block_m * block_k * size_of[a_type]()
            var smem_b = block_n * block_k * size_of[b_type]()
            if smem_a + smem_b <= 32 * 1024:
                block_k *= 2

    var block_tile_shape = Index(block_m, block_n, block_k)
    var warp_tile_shape = block_tile_shape

    # Warp partition block_m and block_n.
    for i in reversed(range(2)):
        if (
            block_tile_shape[i] >= 32
            and block_tile_shape[i] % 32 == 0
            and num_warps < max_num_warps
        ):
            warp_tile_shape[i] = block_tile_shape[i] // 2
            num_warps *= 2

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        warp_tile_shape=warp_tile_shape,
        mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
        num_pipeline_stages=1,
        num_warp_k_partitions=num_warp_k_partitions,
        pdl_level=pdl_level,
    )


fn _amdgpu_matmul_build_block_shape_list[N: Int]() -> List[IndexList[2]]:
    comptime sm_count = GPUInfo.from_name[_accelerator_arch()]().sm_count

    comptime block_sizes_alias = [16, 32, 64, 96, 128, 160, 192, 224, 256]
    comptime len_block_sizes = len(block_sizes_alias)

    var block_sizes = materialize[block_sizes_alias]()
    var emit_block_shape = InlineArray[Bool, len_block_sizes * len_block_sizes](
        fill=False
    )

    @always_inline
    @parameter
    fn process_m(m: Int):
        var best_score = Int.MAX
        var best_idx = 0
        var idx = 0

        for block_m in block_sizes:
            var m_blocks = ceildiv(m, block_m)

            for block_n in block_sizes:
                var n_blocks = ceildiv(N, block_n)

                var total_blocks = m_blocks * n_blocks
                var batch, extra = divmod(total_blocks - 1, sm_count)
                var score = batch * sm_count + (sm_count - extra - 1)

                if score < best_score or (
                    score == best_score and emit_block_shape[idx]
                ):
                    best_score = score
                    best_idx = idx

                idx += 1

        emit_block_shape[best_idx] = True

    for m in range(16, 1024, 16):
        process_m(m)
    for m in range(1024, 8192, 32):
        process_m(m)

    var block_shape_list = List[IndexList[2]]()

    for idx in range(len(emit_block_shape)):
        if not emit_block_shape[idx]:
            continue

        var idx_m, idx_n = divmod(idx, len_block_sizes)

        block_shape_list.append(Index(block_sizes[idx_m], block_sizes[idx_n]))

    return block_shape_list^


@always_inline
fn _matmul_gpu[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    config: Optional[MatmulConfig[a_type, b_type, c_type, transpose_b]] = None,
    pdl_level: PDLLevel = PDLLevel(),
    register_based_epilogue: Bool = True,
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[mut=False, a_type, 2, _, _],
    b: NDBuffer[mut=False, b_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    comptime a_shape = a.shape
    comptime b_shape = b.shape
    comptime c_shape = c.shape
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    logger.info("---- MATMUL GPU execution started ----")
    logger.info("MxNxK: ", m, "x", n, "x", k, sep="")
    logger.info("Data types: A=", a_type, " B=", b_type, " C=", c_type)
    logger.info("Device: ", ctx.name())
    logger.info(
        "Transpose B: ",
        transpose_b,
        " Use Tensor Core: ",
        use_tensor_core,
        sep="",
    )

    comptime matmul_supported_format_nvidia = (
        a_type in (DType.float32, DType.bfloat16)
        and b_type in (DType.float32, DType.bfloat16)
        and c_type in (DType.float32, DType.bfloat16)
    )

    comptime amd_float8_dtypes = (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    ) if ctx.default_device_info == MI355X else (
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
    )

    comptime matmul_supported_format_amd = (
        (a_type == DType.bfloat16 or a_type in amd_float8_dtypes)
        and b_type == a_type
        and c_type in (DType.float32, DType.bfloat16)
        and not has_amd_rdna_gpu_accelerator()
    )

    comptime matmul_supported_format = matmul_supported_format_amd if has_amd_gpu_accelerator() else matmul_supported_format_nvidia

    # Only the H100 version of gemm supports the compute lambda
    # For the other kernels we wrap it around an epilogue lambda instead.
    @parameter
    @always_inline
    @__copy_capture(c)
    fn compute_lambda_wrapper[
        _dtype: DType, _width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_dtype, _width]):
        comptime if elementwise_compute_lambda_fn:
            comptime compute_lambda = elementwise_compute_lambda_fn.value()
            var output = compute_lambda(coords, val)
            constrained[
                output.dtype == c.type,
                "compute epilogue lambda output and c type mismatch",
            ]()
            c.store[alignment = alignment * size_of[c.type]()](
                coords, rebind[SIMD[c.type, _width]](output)
            )

    comptime elementwise_lambda_wrapper = Optional[elementwise_epilogue_type](
        compute_lambda_wrapper
    ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

    # NOTE: k has to be a multiple of BK * num_stages. Hard coded this condition to 128 for now.
    # TODO: Need to find a better dispatch strategy.
    var h100_matmul_cond = (
        materialize[ctx.default_device_info == H100]()
        and n % 8 == 0
        and a_type == DType.bfloat16
    )
    var amdgpu_matmul_cond = has_amd_gpu_accelerator() and n % 4 == 0
    var multi_gemm_cond = (
        (m > 1 or has_amd_gpu_accelerator())
        and (n % 128 == 0 or h100_matmul_cond or amdgpu_matmul_cond)
        and k % 32 == 0
        and k >= 128
    )
    # fmt: off
    # Require Static K, N in A, B, C
    comptime has_static_NK = b_shape.all_known[2]() \
                      and a_shape.has_value[1]() \
                      and c_shape.has_value[1]()

    logger.info("Static shapes available: N=", b_shape.has_value[1](), " K=", a_shape.has_value[1]())
    # fmt: on

    comptime if env_get_bool["MODULE_USE_VENDOR_BLAS", False]():
        logger.info("Executing: Vendor BLAS")
        return matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            config=config,
        ](c, a, b, ctx)

    comptime use_experimental_kernels = Bool(
        env_get_int["USE_EXPERIMENTAL_KERNELS", 0]()
    )

    comptime bf16_or_fp16 = (DType.bfloat16, DType.float16)
    comptime bf16_or_fp16_fp32 = (DType.bfloat16, DType.float16, DType.float32)

    comptime if (
        has_nvidia_gpu_accelerator()
        and ctx.default_device_info.compute > H100.compute
    ):
        return matmul_dispatch_sm100[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_lambda_wrapper=elementwise_lambda_wrapper,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

    comptime if ctx.default_device_info == H100:
        var status = matmul_dispatch_sm90[
            c_type,
            a_type,
            b_type,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            pdl_level=pdl_level,
        ](c, a, b, ctx)

        if status:
            return

    comptime if (
        matmul_supported_format
        and has_accelerator()
        and not has_apple_gpu_accelerator()
        and use_tensor_core
        and has_static_NK
    ):
        if multi_gemm_cond:

            @always_inline
            @parameter
            fn _multistage_gemm[
                config: MatmulConfig[a_type, b_type, c_type, transpose_b]
            ](
                runtime_config: MatmulConfig[
                    a_type, b_type, c_type, transpose_b
                ]
            ) raises:
                return multistage_gemm[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    runtime_config,
                    ctx,
                )

            @always_inline
            @parameter
            fn _multistage_gemm[
                config: MatmulConfig[a_type, b_type, c_type, transpose_b]
            ]() raises:
                comptime if config.num_k_partitions > 1:
                    return _multistage_gemm[config](config)

                return multistage_gemm[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](
                    rebind[NDBuffer[c_type, 2, c.origin, c_shape]](c),
                    rebind[NDBuffer[a_type, 2, a.origin, a_shape]](a),
                    rebind[NDBuffer[b_type, 2, b.origin, b_shape]](b),
                    ctx,
                )

            # Allow caller to overwrite dispatch heuristic with their own config.
            comptime if config:
                return _multistage_gemm[config.value()]()

            comptime static_N = c_shape.get[1]()
            comptime static_K = a_shape.get[1]()

            comptime if has_amd_gpu_accelerator():

                @always_inline
                @parameter
                fn kernel_helper[
                    block_m: Int,
                    block_n: Int,
                    *,
                    num_k_partitions: Int = 1,
                    num_pipeline_stages: Int = 1,
                ]() raises:
                    comptime config = MatmulConfig[
                        a_type, b_type, c_type, transpose_b
                    ](
                        block_tile_shape=Index(
                            block_m, block_n, _bk_base[a_type, True]()
                        ),
                        warp_tile_shape=Index(
                            block_m // 2, block_n // 2, _bk_base[a_type, True]()
                        ),
                        mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
                        num_pipeline_stages=UInt(num_pipeline_stages),
                        num_k_partitions=UInt(num_k_partitions),
                        pdl_level=pdl_level,
                    )
                    return _multistage_gemm[config]()

                if m == 1:
                    return gemv_gpu[
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_wrapper,
                        pdl_level=pdl_level,
                    ](c, a, b, ctx)

                comptime if not transpose_b:
                    return kernel_helper[128, 128, num_pipeline_stages=2]()
                elif env_get_bool["AUTOTUNING_MODE", False]():
                    comptime block_m = env_get_int["TUNE_BM", 128]()
                    comptime block_n = env_get_int["TUNE_BN", 128]()
                    comptime num_k_partitions = env_get_int[
                        "TUNE_NUM_K_PARTITIONS", 1
                    ]()
                    return kernel_helper[
                        block_m, block_n, num_k_partitions=num_k_partitions
                    ]()

                comptime sm_count = ctx.default_device_info.sm_count
                comptime block_shape_list = _amdgpu_matmul_build_block_shape_list[
                    static_N
                ]()

                # Auto-tune block shape selection: Find the configuration that minimizes
                # SM idle time by scoring how evenly work distributes across all SMs.
                # Lower score = better load balance (fewer idle SMs in the last wave).
                var best_idx = 0
                var best_score = Int.MAX

                comptime for i in range(len(block_shape_list)):
                    comptime block_shape = block_shape_list[i]
                    comptime block_m = block_shape[0]
                    comptime block_n = block_shape[1]
                    comptime n_blocks = ceildiv(static_N, block_n)

                    var m_blocks = ceildiv(m, block_m)
                    var total_blocks = m_blocks * n_blocks
                    var batch, extra = divmod(total_blocks - 1, sm_count)
                    var score = batch * sm_count + (sm_count - extra - 1)

                    if score < best_score:
                        best_idx = i
                        best_score = score

                comptime for i in range(len(block_shape_list)):
                    if best_idx == i:
                        comptime config = _amdgpu_matmul_config_from_block_shape[
                            c_type,
                            a_type,
                            b_type,
                            transpose_b,
                            static_K,
                            pdl_level,
                        ](
                            block_shape_list[i]
                        )
                        return _multistage_gemm[config]()

                return kernel_helper[128, 128]()

            else:
                comptime if (
                    a_type == b_type
                    and a_type.is_half_float()
                    and ctx.default_device_info == A100
                    and transpose_b
                ):
                    comptime Ms: List[Int32] = [
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        768,
                        1024,
                        2048,
                        4096,
                    ]
                    try:
                        comptime for M in Ms:
                            if M <= Int32(m):
                                comptime key = String(
                                    M, "_", static_N, "_", static_K
                                )
                                comptime curr_config = create_matmul_configs_ampere[
                                    key, a_type, b_type, c_type, transpose_b
                                ]()
                                if curr_config.num_pipeline_stages == 0:
                                    raise Error("no match for the triple")
                                return _multistage_gemm[curr_config]()
                        raise "no match for the triple"
                    except:
                        pass

                comptime kernels = MatmulKernels[
                    a_type, b_type, c_type, transpose_b
                ]()

                var best_config = select_config[
                    a_type, b_type, c_type, transpose_b
                ](m, n, k, ctx)

                if best_config == kernels.ampere_256x64_4:
                    _multistage_gemm[kernels.ampere_256x64_4](best_config)

                elif best_config == kernels.ampere_256x128_3:
                    _multistage_gemm[kernels.ampere_256x128_3](best_config)

                else:  # Default kernel 128x128_4
                    _multistage_gemm[kernels.ampere_128x128_4](best_config)
                return

    comptime if not a_type.is_float8():
        if n == 1 or m == 1:
            gemv_gpu[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                pdl_level=pdl_level,
            ](c, a, b, ctx)
            return

    comptime vendor_blas_fallback_dtypes = (
        DType.float32,
        DType.float16,
        DType.bfloat16,
    )

    comptime if (
        a_type in vendor_blas_fallback_dtypes
        and b_type in vendor_blas_fallback_dtypes
        and c_type in vendor_blas_fallback_dtypes
        and not has_apple_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        # to disable vendor fallback, run export MODULAR_DISABLE_VENDOR_FALLBACK=1 in the environment
        and not _vendor_blas_fallback_disabled()
    ):
        logger.info("Executing: vendor BLAS fallback")
        try:
            return matmul_vendor[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                config=config,
            ](c, a, b, ctx)
        except:
            # Fallback to the naive kernel.
            logger.warning("Vendor BLAS failed")

    comptime if has_amd_rdna_gpu_accelerator() and a_type in (
        DType.float16,
        DType.bfloat16,
    ):
        if m > 1 and n > 1 and k >= 16 and k % 16 == 0:
            logger.info("Executing: RDNA WMMA MATMUL kernel")
            comptime _BLOCK_M = 64
            comptime _BLOCK_N = 64
            comptime _NUM_WARPS = 4
            comptime _WARP_SIZE = 32

            var c_lt = from_ndbuffer_row_major(c)
            var a_lt = from_ndbuffer_row_major(a)
            var b_lt = from_ndbuffer_row_major(b)

            comptime rdna_kernel = gemm_kernel_rdna[
                c_type,
                a_type,
                b_type,
                c_lt.layout,
                a_lt.layout,
                b_lt.layout,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
            ]

            ctx.enqueue_function[rdna_kernel, rdna_kernel](
                c_lt,
                a_lt,
                b_lt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, _BLOCK_N), ceildiv(m, _BLOCK_M)),
                block_dim=(_NUM_WARPS * _WARP_SIZE,),
            )
            return

    logger.info("Executing: Naive MATMUL kernel")
    comptime BLOCK_DIM = 16

    var c_layout_tensor = from_ndbuffer_row_major(c)
    var a_layout_tensor = from_ndbuffer_row_major(a)
    var b_layout_tensor = from_ndbuffer_row_major(b)

    comptime kernel = matmul_kernel_naive[
        c_type,
        a_type,
        b_type,
        c_layout_tensor.layout,
        a_layout_tensor.layout,
        b_layout_tensor.layout,
        BLOCK_DIM,
        transpose_b,
        elementwise_lambda_fn=elementwise_lambda_wrapper,
    ]

    ctx.enqueue_function[kernel, kernel](
        c_layout_tensor,
        a_layout_tensor,
        b_layout_tensor,
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
    c_layout: Layout,
    work_space_layout: Layout,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[mut=True, c_type, c_layout],
    work_space: LayoutTensor[work_space_type, work_space_layout],
    ctx: DeviceContext,
) raises:
    comptime simd_width = simd_width_of[c_type, target = get_gpu_target()]()
    var num_partitions = work_space.dim[0]()
    var M = c.dim[0]()
    var N = c.dim[1]()

    @always_inline
    @__copy_capture(c, work_space, num_partitions)
    @parameter
    fn _reduce[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](c_coord: IndexList[rank]):
        var idx = Index(0, c_coord[0], c_coord[1])
        var vec = work_space.load[width=simd_width](idx)
        for k in range(1, num_partitions):
            vec += work_space.load[width=simd_width](
                Index(k, c_coord[0], c_coord[1])
            )

        comptime align = align_of[SIMD[c_type, simd_width]]()

        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            epilogue[alignment=align](
                rebind[IndexList[2]](c_coord), vec.cast[c_type]()
            )
        else:
            c.store[width=simd_width](
                c_coord[0], c_coord[1], vec.cast[c_type]()
            )

    elementwise[_reduce, simd_width, target="gpu"](Index(M, N), ctx)


fn multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, c_shape],
    a: NDBuffer[a_type, 2, _, a_shape],
    b: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()

    logger.info("------ Dispatching to Multistage GEMM ------")
    logger.info(config)

    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b = from_ndbuffer_row_major(b)

    comptime if (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        logger.info("Executing: AMD standard GEMM (no split-K)")
        comptime gemm_kernel_type = gemm_kernel_amd[
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

        ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            grid_dim=config.grid_dim(UInt(M), UInt(N)),
            block_dim=config.block_dim(),
        )

    else:
        logger.info("Executing: standard GEMM (no split-K)")
        comptime gemm_kernel_type = multistage_gemm_kernel[
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
        ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            grid_dim=config.grid_dim(UInt(M), UInt(N)),
            block_dim=config.block_dim(),
            shared_mem_bytes=config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )


fn multistage_gemm[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, c_shape],
    a: NDBuffer[mut=False, a_type, 2, _, a_shape],
    b: NDBuffer[mut=False, b_type, 2, _, b_shape],
    runtime_config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()

    logger.info("------ Dispatching to Multistage GEMM ------")
    logger.info(config)
    logger.info("K partitions:", runtime_config.num_k_partitions)

    var tensor_c = from_ndbuffer_row_major(c)
    var tensor_a = from_ndbuffer_row_major(a)
    var tensor_b = from_ndbuffer_row_major(b)

    if runtime_config.num_k_partitions > 1:
        logger.info(
            "Executing: split-K with parallel reduction (workspace-based)"
        )
        comptime work_space_type = config.split_k_reduction_type
        var work_space_data = ctx.enqueue_create_buffer[work_space_type](
            Int(runtime_config.num_k_partitions * UInt(M) * UInt(N))
        )
        comptime static_N = tensor_c.layout.shape[1].value()
        comptime work_space_layout = Layout.row_major(
            UNKNOWN_VALUE, UNKNOWN_VALUE, static_N
        )
        var work_space_runtime_layout = RuntimeLayout[
            work_space_layout
        ].row_major(Index(runtime_config.num_k_partitions, M, N))

        var tensor_work_space = LayoutTensor[
            work_space_type,
            work_space_layout,
            MutAnyOrigin,
        ](work_space_data, work_space_runtime_layout)

        comptime gemm_kernel_type = multistage_gemm_split_k_kernel[
            c_type,
            tensor_c.layout,
            a_type,
            tensor_a.layout,
            b_type,
            tensor_b.layout,
            work_space_type,
            tensor_work_space.layout,
            transpose_b,
            config,
            elementwise_lambda_fn,
        ]

        comptime if has_amd_gpu_accelerator() and not has_amd_rdna_gpu_accelerator():
            ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                tensor_work_space,
                Int(runtime_config.num_k_partitions),
                grid_dim=runtime_config.grid_dim(UInt(M), UInt(N)),
                block_dim=runtime_config.block_dim(),
            )
        else:
            ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                tensor_work_space,
                Int(runtime_config.num_k_partitions),
                grid_dim=runtime_config.grid_dim(UInt(M), UInt(N)),
                block_dim=runtime_config.block_dim(),
                shared_mem_bytes=runtime_config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(runtime_config.shared_mem_usage())
                ),
            )

        split_k_reduce[elementwise_lambda_fn=elementwise_lambda_fn](
            tensor_c, tensor_work_space, ctx
        )

        _ = work_space_data^
        return

    # Dispatch w/o split K
    comptime if (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        logger.info("Executing: AMD standard GEMM (no split-K)")
        comptime gemm_kernel_type = gemm_kernel_amd[
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
        ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            grid_dim=runtime_config.grid_dim(UInt(M), UInt(N)),
            block_dim=runtime_config.block_dim(),
        )

    else:
        logger.info("Executing: standard GEMM (no split-K)")
        comptime gemm_kernel_type = multistage_gemm_kernel[
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

        ctx.enqueue_function[gemm_kernel_type, gemm_kernel_type](
            tensor_c,
            tensor_a,
            tensor_b,
            grid_dim=runtime_config.grid_dim(UInt(M), UInt(N)),
            block_dim=runtime_config.block_dim(),
            shared_mem_bytes=runtime_config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )
