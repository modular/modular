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
from std.math import align_down, ceildiv
from std.sys import (
    align_of,
    get_defined_bool,
    get_defined_int,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)
from std.sys.info import _accelerator_arch, _has_blackwell_tcgen05

from std.algorithm.functional import elementwise, tile_and_unswitch
from std.gpu import (
    WARP_SIZE,
    barrier,
    global_idx,
    thread_idx,
)
from std.gpu.primitives.grid_controls import PDLLevel
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from std.gpu.host.info import A100, B200, H100, MI355X, GPUInfo
from layout import (
    Coord,
    Idx,
    LayoutTensor,
    RowMajorLayout,
    RuntimeInt,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.layout import *
from layout.tensor_core import get_mma_shape
from std.logger import Logger
from std.memory import stack_allocation
from std.utils import Index, IndexList
from std.utils.numerics import get_accum_type
from ...gemv import gemv_gpu, gemm_mma_cpasync
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
from .amd import (
    AMDMatmul,
    AMDPingPongMatmul,
    KernelConfig,
    amd_4wave_matmul,
    amd_4wave_split_k_matmul,
    SplitKWorkspace,
)
from .amd_rdna import gemm_kernel_rdna
from .sm80.dispatch import create_matmul_configs_ampere
from .sm90.dispatch import matmul_dispatch_sm90
from .sm100_structured.default.dispatch import matmul_dispatch_sm100
from .sm100_structured.default.matmul import matmul_sm100_fallback

comptime logger = Logger()


@__name(t"matmul_kernel_{c_type}_{a_type}_{b_type}_{tile_size}", mangle=True)
def matmul_kernel[
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
    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime b_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var a = LayoutTensor[a_type, a_layout, ImmutAnyOrigin](
        a_ptr, RuntimeLayout[a_layout].row_major(Index(m, k))
    )
    var b = LayoutTensor[b_type, b_layout, ImmutAnyOrigin](
        b_ptr, RuntimeLayout[b_layout].row_major(Index(k, n))
    )
    var c = LayoutTensor[c_type, c_layout, MutAnyOrigin](
        c_ptr, RuntimeLayout[c_layout].row_major(Index(m, n))
    )

    # Allocate A, B tile in shared memory.
    var a_shared = stack_allocation[
        tile_size * tile_size,
        a_type,
        address_space=AddressSpace.SHARED,
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        b_type,
        address_space=AddressSpace.SHARED,
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
    def update_tile[full_tile: Bool](offset: Int, end: Int, tile_size: Int):
        # If K is not multiple of tile_size, the last tile contains less than
        # tile_size elements. The thread block needs to take addition bound check
        # when loading elements into shared memory.

        # Load A tile into shared memory.
        var a_val: Scalar[a_type]

        comptime if not full_tile:
            a_val = rebind[Scalar[a_type]](a[row, offset + localCol]) if (
                row < m and offset + localCol < k
            ) else 0.0
        else:
            a_val = (
                rebind[Scalar[a_type]](a[row, offset + localCol]) if row
                < m else 0.0
            )
        a_shared[localRow * tile_size + localCol] = a_val

        # Load B tile into shared memory.
        var b_val: Scalar[b_type]

        comptime if not full_tile:
            b_val = rebind[Scalar[b_type]](b[offset + localRow, col]) if (
                col < n and offset + localRow < k
            ) else 0.0
        else:
            b_val = (
                rebind[Scalar[b_type]](b[offset + localRow, col]) if col
                < n else 0.0
            )
        b_shared[localRow * tile_size + localCol] = b_val

        barrier()

        for kk in range(tile_size):
            result += (
                a_shared[localRow * tile_size + kk].cast[s_type]()
                * b_shared[kk * tile_size + localCol].cast[s_type]()
            )

        barrier()

    tile_and_unswitch[update_tile](0, k, tile_size, K_remainder)

    if row < m and col < n:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(row, col), result.cast[c_type]()
            )
        else:
            c[row, col] = result.cast[c_type]()


@__name(
    t"matmul_kernel_naive_{c_type}_{a_type}_{b_type}_{transpose_b}_{BLOCK_DIM}",
    mangle=True,
)
def matmul_kernel_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout_type: TensorLayout,
    a_layout_type: TensorLayout,
    b_layout_type: TensorLayout,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: TileTensor[c_type, c_layout_type, MutAnyOrigin],
    a: TileTensor[a_type, a_layout_type, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout_type, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert c.flat_rank == 2, "expected 2D tensor for c"
    comptime assert a.flat_rank == 2, "expected 2D tensor for a"
    comptime assert b.flat_rank == 2, "expected 2D tensor for b"

    var x = global_idx.x
    var y = global_idx.y

    if x >= m or y >= n:
        return

    var accum = Scalar[s_type]()

    comptime if transpose_b:
        for i in range(k):
            accum += a[x, i].cast[s_type]() * b[y, i].cast[s_type]()

    else:
        for i in range(k):
            accum += a[x, i].cast[s_type]() * b[i, y].cast[s_type]()

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[x, y] = accum.cast[c_type]()


def _amdgpu_get_mma_shape[dtype: DType, transpose_b: Bool]() -> IndexList[3]:
    comptime if transpose_b and _accelerator_arch() == "amdgpu:gfx950":
        comptime if dtype.is_half_float():
            return Index(16, 16, 32)

    return get_mma_shape[dtype, DType.float32]()


def _amdgpu_matmul_config_from_block_shape[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    K: Int,
    pdl_level: PDLLevel = PDLLevel(),
](block_shape: IndexList[2]) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    comptime max_num_warps: Int = 4

    var block_m = block_shape[0]
    var block_n = block_shape[1]
    var block_k = _bk_base[a_type, True]()
    var num_warps: Int = 1
    var num_warp_k_partitions: Int = 1

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
        # a multiple of that expanded block_k size AND the pipeline prologue
        # (depth 2) still fits: K >= 2 * new_block_k.
        if (K % (block_k * 2)) == 0 and K >= 2 * (block_k * 2):
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


def _amdgpu_matmul_build_block_shape_list[N: Int]() -> List[IndexList[2]]:
    comptime sm_count = GPUInfo.from_name[_accelerator_arch()]().sm_count

    comptime block_sizes_alias = [16, 32, 64, 96, 128, 160, 192, 224, 256]
    comptime len_block_sizes = len(block_sizes_alias)

    var block_sizes = materialize[block_sizes_alias]()
    var emit_block_shape = InlineArray[Bool, len_block_sizes * len_block_sizes](
        fill=False
    )

    @always_inline
    @parameter
    def process_m(m: Int):
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
def _matmul_gpu[
    *,
    use_tensor_core: Bool = False,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    ctx: DeviceContext,
    epilogue_tensor: OptionalReg[
        TileTensor[
            c.dtype,
            RowMajorLayout[RuntimeInt[DType.int64], RuntimeInt[DType.int64]],
            ImmutAnyOrigin,
        ]
    ] = None,
) raises:
    """GPU matmul dispatch entry point. Routes to the appropriate kernel
    based on hardware capabilities and tensor properties.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime assert c.flat_rank == 2, "c must have a non-nested layout"
    comptime assert a.flat_rank == 2, "a must have a non-nested layout"
    comptime assert b.flat_rank == 2, "b must have a non-nested layout"

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype

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
        and c_type in amd_float8_dtypes.concat((DType.float32, DType.bfloat16))
        and not has_amd_rdna_gpu_accelerator()
    )

    comptime matmul_supported_format = matmul_supported_format_amd if has_amd_gpu_accelerator() else matmul_supported_format_nvidia

    # Only the H100 version of gemm supports the compute lambda.
    # For the other kernels we wrap it around an epilogue lambda instead.
    @parameter
    @always_inline
    @__copy_capture(c)
    def compute_lambda_wrapper[
        _dtype: DType, _width: SIMDSize, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_dtype, _width]):
        comptime if elementwise_compute_lambda_fn:
            comptime compute_lambda = elementwise_compute_lambda_fn.value()
            var output = compute_lambda(coords, val)
            comptime assert (
                output.dtype == c_type
            ), "compute epilogue lambda output and c type mismatch"
            c.store_linear[alignment=alignment * size_of[c_type]()](
                coords, rebind[SIMD[c_type, _width]](output)
            )

    comptime elementwise_lambda_wrapper = Optional[elementwise_epilogue_type](
        compute_lambda_wrapper
    ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

    # Helper for gemv_gpu dispatch — passes TileTensor directly.
    @always_inline
    @parameter
    def _gemv_dispatch() raises:
        gemv_gpu[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
            pdl_level=PDLLevel.ON,
        ](c, a, b, ctx)

    # NOTE: k has to be a multiple of BK * num_stages. Hard coded this condition to 128 for now.
    # TODO: Need to find a better dispatch strategy.
    var h100_matmul_cond = (
        materialize[ctx.default_device_info == H100]()
        and n % 8 == 0
        and a_type == DType.bfloat16
    )
    var amdgpu_matmul_cond = has_amd_gpu_accelerator() and n % 4 == 0
    # AMD matmul kernels require K % BK == 0 and K >= 2*BK due to the
    # 2-deep software pipeline prologue.  BK = _bk_base (128 for FP8,
    # 64 for BF16 on AMD).  Use that as the minimum alignment/size gate
    # so unsupported K values fall through to vendor BLAS.
    comptime amd_bk = _bk_base[
        a_type, True
    ]() if has_amd_gpu_accelerator() else 1
    var amd_k_cond = (
        k % amd_bk == 0 and k >= 2 * amd_bk
    ) if has_amd_gpu_accelerator() else True

    var multi_gemm_cond = (
        (m > 1 or has_amd_gpu_accelerator())
        and (n % 128 == 0 or h100_matmul_cond or amdgpu_matmul_cond)
        and k % 32 == 0
        and k >= 128
        and amd_k_cond
    )

    # Static shape queries from TileTensor. -1 means dynamic.
    # fmt: off
    comptime has_static_NK = (b.static_shape[0] > -1 and b.static_shape[1] > -1) \
                      and a.static_shape[1] > -1 \
                      and c.static_shape[1] > -1

    logger.info("Static shapes available: N=", b.static_shape[1] > -1, " K=", a.static_shape[1] > -1)
    # fmt: on

    comptime if get_defined_bool["MODULE_USE_VENDOR_BLAS", False]():
        logger.info("Executing: Vendor BLAS")
        return matmul_vendor[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](c, a, b, ctx)

    comptime if (has_nvidia_gpu_accelerator() and _has_blackwell_tcgen05()):
        return matmul_dispatch_sm100[
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_lambda_wrapper=elementwise_lambda_wrapper,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=PDLLevel.ON,
            has_epilogue_tensor=has_epilogue_tensor,
            epilogue_is_1d=epilogue_is_1d,
        ](c, a, b, ctx, epilogue_tensor=epilogue_tensor)

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
            def _multistage_gemm[
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
                ](c, a, b, runtime_config, ctx)

            @always_inline
            @parameter
            def _multistage_gemm[
                config: MatmulConfig[a_type, b_type, c_type, transpose_b]
            ]() raises:
                comptime if config.num_k_partitions > 1:
                    return _multistage_gemm[config](config)

                return multistage_gemm[
                    transpose_b=transpose_b,
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                ](c, a, b, ctx)

            comptime static_N = c.static_shape[1]
            comptime static_K = a.static_shape[1]

            comptime if has_amd_gpu_accelerator():

                @always_inline
                @parameter
                def kernel_helper[
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
                        num_pipeline_stages=num_pipeline_stages,
                        num_k_partitions=num_k_partitions,
                        pdl_level=pdl_level,
                    )
                    return _multistage_gemm[config]()

                if m == 1:
                    return _gemv_dispatch()

                # NOTE: this branch previously routed nine (N, K) shapes
                # to hipBLASLt via `matmul_vendor` for both BF16 and FP8
                # (the `vendor_blas_NK` / `vendor_blas_NK_m` tables
                # carrying TODO(KERN-2592)). A direct-kernel comparison
                # sweep on MI355X now shows:
                #   - BF16: dispatcher matches vendor within +/-1.8% on
                #     every shape and every M point. The vendor route
                #     was a no-op for BF16.
                #   - FP8 : 4 of the 9 shapes are dispatcher wins at
                #     every M (+7% to +90%); the other 4 had vendor
                #     winning small-M decode by 10..25%, but those
                #     shapes are not in any production model we ship
                #     today (Hippo/Kimi). The single overlap with a
                #     production model (N=2304 K=16384) is a dispatcher
                #     win at every routed M.
                # The vendor table is removed entirely. If a future
                # model needs a small-M wide-N FP8 cell (e.g. M=16
                # N=55296) we can reintroduce a narrow vendor route or
                # land a tile-tuned kernel for that regime.

                comptime if not transpose_b:
                    return kernel_helper[128, 128, num_pipeline_stages=2]()

                # FP8 transpose_b on MI355X: route to the 4-wave kernel
                # family in its bench-validated regime. The 4-wave
                # kernels' fixed BM/BN auto-pick (64 or 128) outperforms
                # the prior auto-tuned `multistage_gemm` dispatch on
                # small-to-medium square shapes (N=K up to 8192) and on
                # skinny-N shapes (N <= 4096 with large K). At very
                # large N=K (>= 16384) and at large M, the existing
                # auto-tuned generic block selection wins, so we fall
                # through to it.
                #
                # Routing inside the gate (derived from the kbench
                # autotune sweep in `tuning_table_mi355_fp8.yaml`,
                # covering Hippo Attn QKV (N=2304) and Kimi Attn out
                # (N=7168) over the production M sweep):
                #
                # Wide-N (4096 <= N <= 8192):
                #   m <=   64          -> amd_4wave_split_k_matmul[4]   BM=BN=64
                #     64 < m <=  128   -> amd_4wave_split_k_matmul[4]   BM=BN=128
                #    128 < m <=  256   -> amd_4wave_split_k_matmul[2]   BM=BN=128
                #    256 < m <= 1024   -> amd_4wave_matmul              BM=BN=128
                #          m >  1024   -> fall through (multistage's
                #                          ping_pong picks up at
                #                          M >= 600, N >= 4096)
                #
                # Skinny-N (N < 4096, e.g. N=2304 K=16384):
                #   m <=  128          -> amd_4wave_split_k_matmul[4]   BM=BN=64
                #    128 < m <=  256   -> amd_4wave_split_k_matmul[4]   BM=BN=128
                #    256 < m <= 2048   -> amd_4wave_split_k_matmul[2]   BM=BN=128
                #          m >  2048   -> fall through
                #
                # Gates:
                #   - 1024 <= static_N <= 8192: lower bound keeps
                #     dispatch off pathologically-narrow N; upper
                #     bound excludes N=K=16384+ where the auto-tuned
                #     dispatch's deeper-tile configs win.
                #   - static_K >= 4096: avoids a 4-wave kernel
                #     pathology at skinny-K (K < 4096) where the
                #     non-split 4-wave can collapse to <100 TFLOPS
                #     on certain (N, K, M) combos.
                #   - static_K % 1024 == 0: required by split-K(4)
                #     since K_per_split = K/4 must be a multiple of
                #     2*BK = 256; production K values are all
                #     1024-aligned.
                comptime if (
                    a_type.is_float8()
                    and transpose_b
                    and ctx.default_device_info == MI355X
                ):
                    comptime if (
                        static_N >= 1024
                        and static_N <= 8192
                        and static_K >= 4096
                        and static_K % 1024 == 0
                    ):
                        # Autotune entry point. When `AUTOTUNING_MODE=True`
                        # AND `TUNE_4WAVE_KERNEL` is set (1=split-K(4),
                        # 2=split-K(2), 3=non-split 4-wave), bypass the
                        # closed-form M cutoffs below and dispatch to the
                        # exact kernel + (BM, BN) the autotune driver
                        # picked. `TUNE_4WAVE_KERNEL=0` (default) keeps
                        # the closed-form heuristic.
                        comptime _tune_4wave = get_defined_int[
                            "TUNE_4WAVE_KERNEL", 0
                        ]()
                        comptime if _tune_4wave != 0:
                            comptime _tune_bm = get_defined_int[
                                "TUNE_4WAVE_BM", 0
                            ]()
                            comptime _tune_bn = get_defined_int[
                                "TUNE_4WAVE_BN", 0
                            ]()
                            comptime _tune_swizzle = get_defined_bool[
                                "TUNE_4WAVE_SWIZZLE", True
                            ]()
                            comptime if _tune_4wave == 1:
                                logger.info(
                                    "Autotune: AMD 4-wave + split-K(4) FP8"
                                    " matmul"
                                )
                                var sk_ws_4 = SplitKWorkspace[4](
                                    ctx, m * static_N
                                )
                                amd_4wave_split_k_matmul[
                                    num_splits=4,
                                    enable_swizzle=_tune_swizzle,
                                    block_m_override=_tune_bm,
                                    block_n_override=_tune_bn,
                                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                                ](a, b, c, ctx, workspace=sk_ws_4)
                                _ = sk_ws_4^
                                return
                            elif _tune_4wave == 2:
                                logger.info(
                                    "Autotune: AMD 4-wave + split-K(2) FP8"
                                    " matmul"
                                )
                                var sk_ws_2 = SplitKWorkspace[2](
                                    ctx, m * static_N
                                )
                                amd_4wave_split_k_matmul[
                                    num_splits=2,
                                    enable_swizzle=_tune_swizzle,
                                    block_m_override=_tune_bm,
                                    block_n_override=_tune_bn,
                                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                                ](a, b, c, ctx, workspace=sk_ws_2)
                                _ = sk_ws_2^
                                return
                            else:
                                logger.info(
                                    "Autotune: AMD 4-wave (no split-K) FP8"
                                    " matmul"
                                )
                                return amd_4wave_matmul[
                                    enable_swizzle=_tune_swizzle,
                                    block_m_override=_tune_bm,
                                    block_n_override=_tune_bn,
                                    elementwise_lambda_fn=elementwise_lambda_wrapper,
                                ](a, b, c, ctx)

                        # The cutoffs and tile shapes below come from the
                        # `tuning_table_mi355_fp8.yaml` autotune sweep on
                        # Hippo Attn QKV (N=2304) and Kimi Attn out
                        # (N=7168). The previous closed-form clamped the
                        # launcher's BM/BN auto-pick to 64x64 for M <=
                        # 512, which left up to +57% on the table at
                        # wide-N M=512 (autotune picks BM=BN=128 there).
                        # The kernel boundaries also shifted: wide-N
                        # split-K(2) only wins at M=129..256, 4-wave
                        # widens to M=257..1024, and skinny-N collapses
                        # 4-wave into split-K(2) all the way to M=2048.
                        #
                        # Wide-N (4096 <= N <= 8192):
                        #   M <=   64           -> split-K(4) 64x64
                        #     64 < M <= 128     -> split-K(4) 128x128
                        #    128 < M <= 256     -> split-K(2) 128x128
                        #    256 < M <= 1024    -> 4-wave     128x128
                        #          M >  1024    -> fall through
                        #
                        # Skinny-N (N < 4096):
                        #   M <=  128           -> split-K(4) 64x64
                        #    128 < M <= 256     -> split-K(4) 128x128
                        #    256 < M <= 2048    -> split-K(2) 128x128
                        #          M >  2048    -> fall through
                        comptime is_skinny_n = static_N < 4096
                        comptime sk4_64_max = 128 if is_skinny_n else 64
                        comptime sk4_128_max = 256 if is_skinny_n else 128
                        comptime sk2_128_max = 2048 if is_skinny_n else 256
                        comptime fwave_max = 0 if is_skinny_n else 1024
                        if m <= sk4_64_max:
                            logger.info(
                                "Executing: AMD 4-wave + split-K(4) FP8 matmul"
                            )
                            var sk_ws_4_64 = SplitKWorkspace[4](
                                ctx, m * static_N
                            )
                            amd_4wave_split_k_matmul[
                                num_splits=4,
                                elementwise_lambda_fn=elementwise_lambda_wrapper,
                            ](a, b, c, ctx, workspace=sk_ws_4_64)
                            _ = sk_ws_4_64^
                            return
                        elif m <= sk4_128_max:
                            logger.info(
                                "Executing: AMD 4-wave + split-K(4) FP8"
                                " matmul (BM=BN=128)"
                            )
                            var sk_ws_4_128 = SplitKWorkspace[4](
                                ctx, m * static_N
                            )
                            amd_4wave_split_k_matmul[
                                num_splits=4,
                                block_m_override=128,
                                block_n_override=128,
                                elementwise_lambda_fn=elementwise_lambda_wrapper,
                            ](a, b, c, ctx, workspace=sk_ws_4_128)
                            _ = sk_ws_4_128^
                            return
                        elif m <= sk2_128_max:
                            logger.info(
                                "Executing: AMD 4-wave + split-K(2) FP8"
                                " matmul (BM=BN=128)"
                            )
                            var sk_ws_2_128 = SplitKWorkspace[2](
                                ctx, m * static_N
                            )
                            amd_4wave_split_k_matmul[
                                num_splits=2,
                                block_m_override=128,
                                block_n_override=128,
                                elementwise_lambda_fn=elementwise_lambda_wrapper,
                            ](a, b, c, ctx, workspace=sk_ws_2_128)
                            _ = sk_ws_2_128^
                            return
                        elif m <= fwave_max:
                            logger.info(
                                "Executing: AMD 4-wave FP8 matmul (BM=BN=128)"
                            )
                            return amd_4wave_matmul[
                                block_m_override=128,
                                block_n_override=128,
                                elementwise_lambda_fn=elementwise_lambda_wrapper,
                            ](a, b, c, ctx)
                        # else: fall through to existing dispatch.

                comptime if get_defined_bool["AUTOTUNING_MODE", False]():
                    comptime block_m = get_defined_int["TUNE_BM", 128]()
                    comptime block_n = get_defined_int["TUNE_BN", 128]()
                    comptime block_k = get_defined_int[
                        "TUNE_BK", _bk_base[a_type, True]()
                    ]()
                    comptime num_k_partitions = get_defined_int[
                        "TUNE_NUM_K_PARTITIONS", 1
                    ]()
                    comptime config = MatmulConfig[
                        a_type, b_type, c_type, transpose_b
                    ](
                        block_tile_shape=Index(block_m, block_n, block_k),
                        warp_tile_shape=Index(
                            block_m // 2, block_n // 2, block_k
                        ),
                        mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
                        num_pipeline_stages=1,
                        num_k_partitions=num_k_partitions,
                        pdl_level=pdl_level,
                    )
                    return _multistage_gemm[config]()

                # Shape-specific FP8 configs for small M (128-256).
                # These match hipBLASLt's tile choices which use deeper K
                # tiles and adapted block shapes for better per-block
                # throughput at low CU occupancy.
                # Format: Index(N, K, BM, BN, BK)
                # Small-M FP8 dispatch (M=128-256): use autotuned block
                # shapes that outperform the generic auto-tuner.
                # Derived from sweep over BM/BN/BK on MI355X.
                comptime if a_type.is_float8() and transpose_b:
                    if m >= 128 and m <= 256:

                        @always_inline
                        @parameter
                        def _small_m_gemm[
                            _bm: Int, _bn: Int, _bk: Int
                        ]() raises:
                            comptime config = MatmulConfig[
                                a_type, b_type, c_type, transpose_b
                            ](
                                block_tile_shape=Index(_bm, _bn, _bk),
                                warp_tile_shape=Index(_bm // 2, _bn // 2, _bk),
                                mma_shape=_amdgpu_get_mma_shape[
                                    a_type, transpose_b
                                ](),
                                num_pipeline_stages=1,
                                pdl_level=pdl_level,
                            )
                            return _multistage_gemm[config]()

                        comptime if static_N < 4096:
                            # Narrow N (e.g. N=2304): deep BK=512,
                            # square blocks for balanced compute.
                            # Guard: K must be >= BK to avoid OOB reads.
                            if k >= 512:
                                if m <= 160:
                                    return _small_m_gemm[32, 64, 512]()
                                else:
                                    return _small_m_gemm[64, 64, 512]()
                        else:
                            if m <= 150:
                                # Small M with wide N: BM=64 for
                                # occupancy, BN=128 for N-coverage
                                if k >= 256:
                                    return _small_m_gemm[64, 128, 256]()
                            else:
                                # Larger M (200-256): square 128x128
                                # tiles with BK=256
                                if k >= 256:
                                    return _small_m_gemm[128, 128, 256]()

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
            _gemv_dispatch()
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
            ](c, a, b, ctx)
        except:
            # Fallback to the naive kernel.
            logger.warning("Vendor BLAS failed")

    comptime if has_amd_rdna_gpu_accelerator() and a_type in (
        DType.float16,
        DType.bfloat16,
    ):

        @parameter
        @always_inline
        def _enqueue_rdna_kernel[
            BLOCK_K: Int,
            BLOCK_M: Int,
            BLOCK_N: Int,
            WARPS_M: Int,
            WARPS_N: Int,
            WARP_TILE_M: Int,
            WARP_TILE_N: Int,
        ]() raises:
            comptime NUM_WARPS = WARPS_M * WARPS_N
            comptime rdna_kernel = gemm_kernel_rdna[
                c_type,
                a_type,
                b_type,
                type_of(c).LayoutType,
                type_of(a).LayoutType,
                type_of(b).LayoutType,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                BLOCK_K=BLOCK_K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                WARPS_M=WARPS_M,
                WARPS_N=WARPS_N,
                WARP_TILE_M=WARP_TILE_M,
                WARP_TILE_N=WARP_TILE_N,
            ]

            ctx.enqueue_function[rdna_kernel](
                c,
                a,
                b,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BLOCK_N), ceildiv(m, BLOCK_M)),
                block_dim=(NUM_WARPS * WARP_SIZE,),
            )

        # Large shapes with BK=32: doubles compute per load, halves iterations.
        if m >= 128 and n >= 128 and k >= 32 and k % 32 == 0:
            logger.info("Executing: RDNA WMMA MATMUL kernel (128x128, BK=32)")
            _enqueue_rdna_kernel[
                BLOCK_K=32,
                BLOCK_M=128,
                BLOCK_N=128,
                WARPS_M=8,
                WARPS_N=2,
                WARP_TILE_M=1,
                WARP_TILE_N=4,
            ]()
            return

        # Large shapes with BK=16: fallback for K not divisible by 32.
        if m >= 128 and n >= 128 and k >= 16 and k % 16 == 0:
            logger.info("Executing: RDNA WMMA MATMUL kernel (128x128, BK=16)")
            _enqueue_rdna_kernel[
                BLOCK_K=16,
                BLOCK_M=128,
                BLOCK_N=128,
                WARPS_M=8,
                WARPS_N=2,
                WARP_TILE_M=1,
                WARP_TILE_N=4,
            ]()
            return

        # Moderate shapes: 64x64 tile, 4 warps (2x2), warp_tile 2x2, BK=16.
        if m > 1 and n > 1 and k >= 16 and k % 16 == 0:
            logger.info("Executing: RDNA WMMA MATMUL kernel (64x64)")
            _enqueue_rdna_kernel[
                BLOCK_K=16,
                BLOCK_M=64,
                BLOCK_N=64,
                WARPS_M=2,
                WARPS_N=2,
                WARP_TILE_M=2,
                WARP_TILE_N=2,
            ]()
            return

    logger.info("Executing: Naive MATMUL kernel")
    comptime BLOCK_DIM = 16

    comptime kernel = matmul_kernel_naive[
        c_type,
        a_type,
        b_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        BLOCK_DIM,
        transpose_b,
        elementwise_lambda_fn=elementwise_lambda_wrapper,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        m,
        n,
        k,
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )


@always_inline
def split_k_reduce[
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    work_space: TileTensor,
    ctx: DeviceContext,
) raises:
    comptime c_type = c.dtype
    comptime simd_width = simd_width_of[c_type, target=get_gpu_target()]()
    var c_lt = c.to_layout_tensor()
    var ws_lt = work_space.to_layout_tensor()
    var num_partitions = ws_lt.dim[0]()
    var M = c_lt.dim[0]()
    var N = c_lt.dim[1]()

    @always_inline
    @__copy_capture(c_lt, ws_lt, num_partitions)
    @parameter
    def _reduce[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](c_coord: IndexList[rank]):
        var idx = Index(0, c_coord[0], c_coord[1])
        var vec = ws_lt.load[width=simd_width](idx)
        for k in range(1, num_partitions):
            vec += ws_lt.load[width=simd_width](
                Index(k, c_coord[0], c_coord[1])
            )

        comptime align = align_of[SIMD[c_type, simd_width]]()

        comptime if elementwise_lambda_fn:
            comptime epilogue = elementwise_lambda_fn.value()
            epilogue[alignment=align](
                rebind[IndexList[2]](c_coord), vec.cast[c_type]()
            )
        else:
            c_lt.store[width=simd_width](
                c_coord[0], c_coord[1], vec.cast[c_type]()
            )

    elementwise[_reduce, simd_width, target="gpu"](Index(M, N), ctx)


def multistage_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    ctx: DeviceContext,
) raises:
    """TileTensor overload of `multistage_gemm`. Converts to LayoutTensor and
    dispatches to the appropriate GEMM kernel."""
    var tensor_c = c.to_layout_tensor()
    var tensor_a = a.to_layout_tensor()
    var tensor_b = b.to_layout_tensor()
    comptime a_layout = tensor_a.layout
    comptime b_layout = tensor_b.layout
    _ = tensor_a
    _ = tensor_b

    var M = tensor_c.dim[0]()
    var N = tensor_c.dim[1]()

    logger.info("------ Dispatching to Multistage GEMM ------")
    logger.info(config)

    comptime if (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        comptime if a_type.is_float8():
            # FP8 dispatch: ping-pong for large shapes, standard for small.
            comptime pingpong_config = KernelConfig(
                block_shape=Index(256, 256, 128),
                warp_shape=Index(128, 64, 128),
                mma_shape=Index(16, 16, 128),
            )
            comptime skinny_config = KernelConfig(
                block_shape=Index(128, 256, 128),
                warp_shape=Index(64, 64, 128),
                mma_shape=Index(16, 16, 128),
            )

            @parameter
            @always_inline
            def _launch_pingpong() raises:
                var pp_grid = (
                    ceildiv(N, pingpong_config.block_shape[1]),
                    ceildiv(M, pingpong_config.block_shape[0]),
                )
                var pp_threads = pingpong_config.num_threads()
                comptime k = AMDPingPongMatmul[
                    a_type,
                    b_type,
                    c_type,
                    pingpong_config,
                    enable_swizzle=True,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ].run[a.LayoutType, b.LayoutType, c.LayoutType]
                ctx.enqueue_function[k](
                    a,
                    b,
                    c,
                    grid_dim=pp_grid,
                    block_dim=pp_threads,
                )

            @parameter
            @always_inline
            def _launch_skinny() raises:
                var sk_grid = (
                    ceildiv(N, skinny_config.block_shape[1]),
                    ceildiv(M, skinny_config.block_shape[0]),
                )
                var sk_threads = skinny_config.num_threads()
                comptime k = AMDPingPongMatmul[
                    a_type,
                    b_type,
                    c_type,
                    skinny_config,
                    enable_swizzle=True,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ].run[a.LayoutType, b.LayoutType, c.LayoutType]
                ctx.enqueue_function[k](
                    a,
                    b,
                    c,
                    grid_dim=sk_grid,
                    block_dim=sk_threads,
                )

            @parameter
            @always_inline
            def _launch_standard() raises:
                comptime std_config = MatmulConfig[
                    a_type, b_type, c_type, True
                ](
                    block_tile_shape=config.block_tile_shape,
                    warp_tile_shape=config.warp_tile_shape,
                    mma_shape=config.mma_shape,
                    num_pipeline_stages=config.num_pipeline_stages,
                )
                comptime k = AMDMatmul[
                    a_type,
                    b_type,
                    c_type,
                    True,
                    std_config,
                    elementwise_lambda_fn,
                ].run[c.LayoutType, a.LayoutType, b.LayoutType]
                ctx.enqueue_function[k](
                    c,
                    a,
                    b,
                    grid_dim=std_config.grid_dim(M, N),
                    block_dim=std_config.block_dim(),
                )

            # Dispatch heuristic (from Llama3-405B TP=4 benchmarks).
            if N >= 4096:
                if M >= 600:
                    _launch_pingpong()
                elif M >= 256:
                    _launch_skinny()
                else:
                    _launch_standard()
            else:
                if M >= 750:
                    _launch_skinny()
                else:
                    _launch_standard()
        else:
            # BF16 dispatch.
            comptime bf16_config = MatmulConfig[a_type, b_type, c_type, True](
                block_tile_shape=config.block_tile_shape,
                warp_tile_shape=config.warp_tile_shape,
                mma_shape=config.mma_shape,
                num_pipeline_stages=config.num_pipeline_stages,
                num_warp_k_partitions=config.num_warp_k_partitions,
            )
            comptime k = AMDMatmul[
                a_type,
                b_type,
                c_type,
                True,
                bf16_config,
                elementwise_lambda_fn,
            ].run[c.LayoutType, a.LayoutType, b.LayoutType]
            ctx.enqueue_function[k](
                c,
                a,
                b,
                grid_dim=bf16_config.grid_dim(M, N),
                block_dim=bf16_config.block_dim(),
            )

    else:
        logger.info("Executing: standard GEMM (no split-K)")
        comptime gemm_kernel_type = multistage_gemm_kernel[
            CLT=c.LayoutType,
            ALT=a.LayoutType,
            BLT=b.LayoutType,
            c_linear_idx_type=c.linear_idx_type,
            a_linear_idx_type=a.linear_idx_type,
            b_linear_idx_type=b.linear_idx_type,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function[gemm_kernel_type](
            c,
            a,
            b,
            grid_dim=config.grid_dim(M, N),
            block_dim=config.block_dim(),
            shared_mem_bytes=config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )


def multistage_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    runtime_config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    ctx: DeviceContext,
) raises:
    """TileTensor overload of `multistage_gemm` with runtime config.
    Constrains c to mut=True because `split_k_reduce` requires a mutable
    output tensor."""
    var tensor_c = c.to_layout_tensor()
    var tensor_a = a.to_layout_tensor()
    var tensor_b = b.to_layout_tensor()

    var M = tensor_c.dim[0]()
    var N = tensor_c.dim[1]()

    logger.info("------ Dispatching to Multistage GEMM ------")
    logger.info(config)
    logger.info("K partitions:", runtime_config.num_k_partitions)

    if runtime_config.num_k_partitions > 1:
        logger.info(
            "Executing: split-K with parallel reduction (workspace-based)"
        )
        comptime work_space_type = config.split_k_reduction_type
        var work_space_data = ctx.enqueue_create_buffer[work_space_type](
            runtime_config.num_k_partitions * M * N
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
            ctx.enqueue_function[gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                tensor_work_space,
                runtime_config.num_k_partitions,
                grid_dim=runtime_config.grid_dim(M, N),
                block_dim=runtime_config.block_dim(),
            )
        else:
            ctx.enqueue_function[gemm_kernel_type](
                tensor_c,
                tensor_a,
                tensor_b,
                tensor_work_space,
                runtime_config.num_k_partitions,
                grid_dim=runtime_config.grid_dim(M, N),
                block_dim=runtime_config.block_dim(),
                shared_mem_bytes=runtime_config.shared_mem_usage(),
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(runtime_config.shared_mem_usage())
                ),
            )

        var tt_work_space = TileTensor(
            work_space_data,
            row_major(
                Coord(
                    Idx(runtime_config.num_k_partitions),
                    Idx(M),
                    Idx(N),
                )
            ),
        )
        split_k_reduce[elementwise_lambda_fn=elementwise_lambda_fn](
            c, tt_work_space, ctx
        )

        _ = work_space_data^
        return

    # Dispatch w/o split K
    comptime if (
        has_amd_gpu_accelerator()
        and not has_amd_rdna_gpu_accelerator()
        and transpose_b
    ):
        comptime if a_type.is_float8():
            # FP8 dispatch: ping-pong for large shapes, standard for small.
            comptime pingpong_config = KernelConfig(
                block_shape=Index(256, 256, 128),
                warp_shape=Index(128, 64, 128),
                mma_shape=Index(16, 16, 128),
            )
            comptime skinny_config = KernelConfig(
                block_shape=Index(128, 256, 128),
                warp_shape=Index(64, 64, 128),
                mma_shape=Index(16, 16, 128),
            )

            @parameter
            @always_inline
            def _launch_pingpong() raises:
                var pp_grid = (
                    ceildiv(N, pingpong_config.block_shape[1]),
                    ceildiv(M, pingpong_config.block_shape[0]),
                )
                var pp_threads = pingpong_config.num_threads()
                comptime k = AMDPingPongMatmul[
                    a_type,
                    b_type,
                    c_type,
                    pingpong_config,
                    enable_swizzle=True,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ].run[a.LayoutType, b.LayoutType, c.LayoutType]
                ctx.enqueue_function[k](
                    a,
                    b,
                    c,
                    grid_dim=pp_grid,
                    block_dim=pp_threads,
                )

            @parameter
            @always_inline
            def _launch_skinny() raises:
                var sk_grid = (
                    ceildiv(N, skinny_config.block_shape[1]),
                    ceildiv(M, skinny_config.block_shape[0]),
                )
                var sk_threads = skinny_config.num_threads()
                comptime k = AMDPingPongMatmul[
                    a_type,
                    b_type,
                    c_type,
                    skinny_config,
                    enable_swizzle=True,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ].run[a.LayoutType, b.LayoutType, c.LayoutType]
                ctx.enqueue_function[k](
                    a,
                    b,
                    c,
                    grid_dim=sk_grid,
                    block_dim=sk_threads,
                )

            @parameter
            @always_inline
            def _launch_standard() raises:
                comptime std_config = MatmulConfig[
                    a_type, b_type, c_type, True
                ](
                    block_tile_shape=config.block_tile_shape,
                    warp_tile_shape=config.warp_tile_shape,
                    mma_shape=config.mma_shape,
                    num_pipeline_stages=config.num_pipeline_stages,
                )
                comptime k = AMDMatmul[
                    a_type,
                    b_type,
                    c_type,
                    True,
                    std_config,
                    elementwise_lambda_fn,
                ].run[c.LayoutType, a.LayoutType, b.LayoutType]
                ctx.enqueue_function[k](
                    c,
                    a,
                    b,
                    grid_dim=std_config.grid_dim(M, N),
                    block_dim=std_config.block_dim(),
                )

            # Dispatch heuristic (from Llama3-405B TP=4 benchmarks).
            if N >= 4096:
                if M >= 600:
                    _launch_pingpong()
                elif M >= 256:
                    _launch_skinny()
                else:
                    _launch_standard()
            else:
                if M >= 750:
                    _launch_skinny()
                else:
                    _launch_standard()
        else:
            # BF16 dispatch.
            comptime bf16_config = MatmulConfig[a_type, b_type, c_type, True](
                block_tile_shape=config.block_tile_shape,
                warp_tile_shape=config.warp_tile_shape,
                mma_shape=config.mma_shape,
                num_pipeline_stages=config.num_pipeline_stages,
                num_warp_k_partitions=config.num_warp_k_partitions,
            )
            comptime k = AMDMatmul[
                a_type,
                b_type,
                c_type,
                True,
                bf16_config,
                elementwise_lambda_fn,
            ].run[c.LayoutType, a.LayoutType, b.LayoutType]
            ctx.enqueue_function[k](
                c,
                a,
                b,
                grid_dim=bf16_config.grid_dim(M, N),
                block_dim=bf16_config.block_dim(),
            )

    else:
        logger.info("Executing: standard GEMM (no split-K)")
        comptime gemm_kernel_type = multistage_gemm_kernel[
            CLT=c.LayoutType,
            ALT=a.LayoutType,
            BLT=b.LayoutType,
            c_linear_idx_type=c.linear_idx_type,
            a_linear_idx_type=a.linear_idx_type,
            b_linear_idx_type=b.linear_idx_type,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]

        ctx.enqueue_function[gemm_kernel_type](
            c,
            a,
            b,
            grid_dim=runtime_config.grid_dim(M, N),
            block_dim=runtime_config.block_dim(),
            shared_mem_bytes=runtime_config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )
