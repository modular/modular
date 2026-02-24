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

from math import align_up, ceildiv

from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from layout import Layout, LayoutTensor
from logger import Logger
from gpu.primitives.warp import shuffle_xor
from math import recip
from linalg.fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_f4e2m1x2_to_fp16x2,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
    get_scale_factor,
)
from gpu.host.info import B200
from utils import StaticTuple
from collections import Optional
from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)
from utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from buffer import Dim, NDBuffer
from layout._ndbuffer_stub import from_ndbuffer_row_major
from memory import bitcast
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import barrier
from sys import size_of, align_of, simd_width_of
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from algorithm import elementwise
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_nvfp4,
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


fn heuristic_and_outliers_dispatch[
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
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[scales_dtype, sfa_layout, MutAnyOrigin],
    b_scales: LayoutTensor[scales_dtype, sfb_layout, MutAnyOrigin],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises -> Int:
    var m = c.dim(0)

    comptime static_N = c_layout.shape[1].value()
    comptime static_K = a_layout.shape[
        1
    ].value() * 2 if a_type == DType.uint8 else a_layout.shape[1].value()

    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        (a_type == b_type == DType.uint8)
        and scales_dtype == NVFP4_SF_DTYPE
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    ) or (
        (a_type == b_type == DType.float8_e4m3fn)
        and scales_dtype == MXFP8_SF_DTYPE
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    ), (
        "Only support NVFP4_SF_DTYPE (float8_e4m3fn) or MXFP8_SF_DTYPE"
        " (float8_e8m0fnu) for scales for now."
    )

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

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())

    comptime outliers = Table(
        _get_tuning_list_sm100_nvfp4(), "nvfp4_heuristic_outliers"
    ) if a_type == DType.uint8 else Table(
        _get_tuning_list_sm100_mxfp8(), "mxfp8_heuristic_outliers"
    )

    comptime scaling_kind = UMMAKind.KIND_MXF4NVF4 if a_type == DType.uint8 else UMMAKind.KIND_MXF8F6F4

    @parameter
    @always_inline
    fn rule(x: TuningConfigSM100) -> Bool:
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
                block_swizzle_size=Int(tuning_config.block_swizzle_size),
                raster_order=tuning_config.rasterize_order,
                AB_swapped=tuning_config.swapAB,
                num_accum_pipeline_stages=Int(
                    tuning_config.num_accum_pipeline_stages
                ),
                num_clc_pipeline_stages=Int(
                    tuning_config.num_clc_pipeline_stages
                ),
                k_group_size=Int(tuning_config.k_group_size),
                num_split_k=tuning_config.num_split_k,
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


fn _block_scaled_matmul_with_epilogue[
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
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[scales_dtype, sfa_layout, MutAnyOrigin],
    b_scales: LayoutTensor[scales_dtype, sfb_layout, MutAnyOrigin],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Our sm100 block scaled matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 block scaled matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations.
    """

    var m = c.dim(0)
    var n = c.dim(1)
    if m == 0 or n == 0:
        return

    comptime if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated!"

        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
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
            simd_width_of[c_type, target = get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
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

            blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                transpose_b=transpose_b,
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
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](c.size())
        var c_tmp = c
        c_tmp.ptr = tmp_device_buffer.unsafe_ptr()

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


fn _vendor_blas_block_scaled_matmul_with_epilogue[
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
    comptime assert (
        ctx.default_device_info.compute == B200.compute
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
            simd_width_of[c_type, target = get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
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
