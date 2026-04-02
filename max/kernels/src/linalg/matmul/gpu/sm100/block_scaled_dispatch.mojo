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


from std.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, Idx, Layout, LayoutTensor, TileTensor, row_major
from std.logger import Logger
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    get_scaling_kind,
)
from std.gpu.host.info import _is_sm10x_gpu
from std.collections import Optional
from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)
from std.utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from std.memory import UnsafePointer
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.sys import size_of, simd_width_of
from std.algorithm import elementwise
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_nvfp4,
    _get_tuning_list_sm100_mxfp4,
    _get_tuning_list_sm100_mxfp8,
)
from internal_utils import Table
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    build_block_scaled_configs,
    choose_block_scaled_config,
)

comptime logger = Logger()

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises -> Int:
    var m = Int(c.dim[0]())

    comptime scaling_kind = get_scaling_kind[
        a_type, scales_dtype, SF_VECTOR_SIZE
    ]()
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )

    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1] * 2 if is_fp4 else a.static_shape[1]

    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        (
            scaling_kind == UMMAKind.KIND_MXF4NVF4
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF4
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF8F6F4
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        )
    ), "Only support NVFP4, MXFP4, or MXFP8 scale/dtype combinations."

    comptime assert (
        a_scales.static_shape[1] == b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime outliers = Table(
        _get_tuning_list_sm100_nvfp4(), "nvfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4NVF4 else Table(
        _get_tuning_list_sm100_mxfp4(), "mxfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4 else Table(
        _get_tuning_list_sm100_mxfp8(), "mxfp8_heuristic_outliers"
    )

    @parameter
    @always_inline
    def rule(x: TuningConfigSM100) -> Bool:
        return x.K == static_K and x.N == static_N

    comptime outlier_configs = outliers.find[rule]()

    comptime for tuning_config in outlier_configs:
        if m >= tuning_config.M and m < tuning_config.M_end:
            comptime matmul_config = BlockScaledMatmulConfig[
                a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
            ](
                scaling_kind=scaling_kind,
                mma_shape=tuning_config.mma_shape,
                cta_group=tuning_config.cta_group,
                cluster_shape=tuning_config.cluster_shape,
                block_swizzle_size=tuning_config.block_swizzle_size,
                raster_order=tuning_config.rasterize_order,
                AB_swapped=tuning_config.swapAB,
                num_accum_pipeline_stages=tuning_config.num_accum_pipeline_stages,
                num_clc_pipeline_stages=tuning_config.num_clc_pipeline_stages,
                k_group_size=tuning_config.k_group_size,
                num_split_k=tuning_config.num_split_k,
                is_small_bn=tuning_config.is_small_bn,
            )

            logger.info("Using tuning config: ", matmul_config)

            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=matmul_config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, a_scales, b_scales, tensor_sf, ctx)

            return DISPATCH_HIT

    # disaptch to small-BN kernel for m == 1 as it's optimized for GEMVs
    if m == 1:
        comptime config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=scaling_kind,
            cta_group=1,
            mma_shape=Index(128, 8, 32),
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            num_accum_pipeline_stages=1,
            k_group_size=2,
            num_clc_pipeline_stages=0,
            AB_swapped=True,
            is_small_bn=True,
        )
        _block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ](c, a, b, a_scales, b_scales, tensor_sf, ctx)

        logger.info("Using small-BN config: ", config)
        return DISPATCH_HIT

    comptime configs = build_block_scaled_configs[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        scales_dtype,
        static_N,
        static_K,
        transpose_b,
    ]()
    var config_runtime = choose_block_scaled_config[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](m, static_N, static_K)

    comptime for config in configs:
        if config_runtime == config:
            logger.info("Using heuristic config: ", config)
            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=config,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
            ](c, a, b, a_scales, b_scales, tensor_sf, ctx)
            return DISPATCH_HIT

    return DISPATCH_MISS


########################################################
# SM100 Block Scaled matmul with normal epilogue kernel dispatch
########################################################


def _block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Our sm100 block scaled matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 block scaled matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations.
    """

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    if m == 0 or n == 0:
        return

    comptime if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated!"

        comptime K_phys = a.static_shape[1]
        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            K=K_phys,
            config=config,
            pdl_level=pdl_level,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            ctx,
            alpha=tensor_sf,
        )
        return
    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # Nvidia GPUs >= sm_100 arch support 32B load/store to global memory.
        comptime use_32b_simd = True
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        # The epilogue lambda takes IndexList[2]. We load from c's raw pointer
        # using row-major offset since TileTensor.load's Coord constraint
        # can't be proved when c's layout type is fully inferred.
        @parameter
        @__copy_capture(c, n)
        def epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = rebind[SIMD[c_type, simd_width]](
                c.load[width=simd_width](Coord(c_coord))
            )
            epilogue[c_type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the sm100 blockwise scaled fp8 matmul and
        # apply the epilogue.
        if c.ptr:
            comptime K_phys = a.static_shape[1]
            blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
                K=K_phys,
                config=config,
                pdl_level=pdl_level,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                ctx,
                alpha=tensor_sf,
            )
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var num_elems = m * n
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](num_elems)
        var c_tmp = TileTensor(
            rebind[UnsafePointer[Scalar[c_type], MutExternalOrigin]](
                tmp_device_buffer.unsafe_ptr()
            ),
            row_major(Coord(Idx(m), Idx(n))),
        )

        _block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            config=config,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tmp,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
        )

        _ = tmp_device_buffer^


def _vendor_blas_block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[scales_dtype, sfa_layout, MutAnyOrigin],
    b_scales: LayoutTensor[scales_dtype, sfb_layout, MutAnyOrigin],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "Only support NVFP4_SF_DTYPE (float8_e4m3fn) for scales for now."

    comptime assert SF_VECTOR_SIZE in (
        NVFP4_SF_VECTOR_SIZE,
    ), "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4)"

    comptime assert (
        sfa_layout.shape[1].value() == sfb_layout.shape[1].value()
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        sfa_layout.shape[2].value()
        == sfb_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        sfa_layout.shape[3].value()
        == sfb_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        sfa_layout.shape[4].value() == sfb_layout.shape[4].value() == SF_ATOM_K
    ), ""

    var m = c.dim(0)
    var n = c.dim(1)
    if m == 0 or n == 0:
        return

    comptime if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated!"

        matmul(
            ctx,
            c,
            a,
            b,
            a_scales=a_scales.get_immutable(),
            b_scales=b_scales.get_immutable(),
            transpose_b=True,
            c_row_major=True,
            alpha=tensor_sf,
        )
    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # Nvidia GPUs >= sm_100 arch support 32B load/store to global memory.
        comptime use_32b_simd = True
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        def epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[width=simd_width,](c_coord)
            epilogue[c_type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the sm100 blockwise scaled fp8 matmul and
        # apply the epilogue.
        if c.ptr:
            var m = c.dim[0]()
            var n = c.dim[1]()

            matmul(
                ctx,
                c,
                a,
                b,
                a_scales=a_scales.get_immutable(),
                b_scales=b_scales.get_immutable(),
                alpha=tensor_sf,
                transpose_b=True,
                c_row_major=True,
            )
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](c.size())
        var c_tmp = c
        c_tmp.ptr = tmp_device_buffer.unsafe_ptr()

        _vendor_blas_block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tmp,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
        )

        _ = tmp_device_buffer^
