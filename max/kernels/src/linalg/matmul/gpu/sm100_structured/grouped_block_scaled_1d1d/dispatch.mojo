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
"""Dispatch logic for grouped 1D-1D block-scaled SM100 matmul.

Selects optimal kernel configuration based on (N, K) shape and workload
size, with parameters tuned via ablation on B200. NVFP4 gets shape-tuned
three-regime dispatch; MXFP4 and MXFP8 use default configs.

When `override=True`, uses the caller's AB_swapped/mma_bn/cta_group/
num_pipeline_stages directly (for ablation studies and benchmarking).
When `override=False` (default), ignores those parameters and selects
from the tuning table based on (N, K) and `estimated_total_m`.

NVFP4 three regimes keyed on avg_m = estimated_total_m / num_active_experts:

  Decode (avg_m <= 8):
  - N=4096, K=7168:  AB_swapped=True, mma_bn=8, cta_group=1, stages=6
  - N=7168, K=2048:  AB_swapped=True, mma_bn=8, cta_group=1, stages=4
  - Default:         AB_swapped=True, mma_bn=8, cta_group=1, stages=auto

  Small prefill (8 < avg_m <= 64):
  - N=4096, K=7168:  AB_swapped=True, mma_bn=64, cta_group=2, stages=6
  - N=7168, K=2048:  AB_swapped=True, mma_bn=64, cta_group=2, stages=6
  - Default:         AB_swapped=True, mma_bn=64, cta_group=2, stages=auto

  Large prefill (avg_m > 64):
  - N=4096, K=7168:  AB_swapped=True, mma_bn=128, cta_group=2, stages=7
  - N=7168, K=2048:  AB_swapped=True, mma_bn=128, cta_group=2, stages=6
  - Default:         AB_swapped=True, mma_bn=128, cta_group=2, stages=auto
"""

from std.collections import Optional

from std.gpu.host import DeviceContext
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.utils.index import Index
from layout import TileTensor

from linalg.fp4_utils import NVFP4_SF_DTYPE, MXFP4_SF_DTYPE, MXFP8_SF_DTYPE
from ..structured_kernels.config import BlockScaledMatmulConfig, GEMMKind
from .grouped_1d1d_matmul import grouped_matmul_block_scaled


def _scaling_kind[a_type: DType, scales_dtype: DType]() -> UMMAKind:
    """Infer UMMAKind from input and scale dtypes."""
    comptime if a_type == DType.uint8 and scales_dtype == NVFP4_SF_DTYPE:
        return UMMAKind.KIND_MXF4NVF4
    elif a_type == DType.uint8 and scales_dtype == MXFP4_SF_DTYPE:
        return UMMAKind.KIND_MXF4
    else:
        comptime assert (
            a_type == DType.float8_e4m3fn and scales_dtype == MXFP8_SF_DTYPE
        ), "unsupported a_type/scales_dtype for grouped block-scaled matmul"
        return UMMAKind.KIND_MXF8F6F4


def _launch_grouped_block_scaled[
    transpose_b: Bool,
    AB_swapped: Bool,
    mma_bn: Int,
    cta_group: Int,
    num_pipeline_stages: Optional[Int] = None,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF4NVF4,
](
    c: TileTensor[...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_scales: TileTensor[...],
    b_scales: TileTensor[...],
    a_offsets: TileTensor[...],
    a_scale_offsets: TileTensor[...],
    expert_ids: TileTensor[...],
    expert_scales: TileTensor[...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Build config and launch grouped block-scaled matmul kernel.

    Parameters:
        transpose_b: Whether B is transposed.
        AB_swapped: Whether A/B operands are swapped.
        mma_bn: MMA tile N dimension.
        cta_group: CTA group size.
        num_pipeline_stages: Pipeline depth override. None = auto-compute.
        scaling_kind: Block-scaling format (NVFP4, MXFP4, or MXFP8).
    """
    comptime a_type = a.dtype
    comptime b_type = b.dtype
    comptime c_type = c.dtype
    comptime sfa_dtype = a_scales.dtype
    comptime sfb_dtype = b_scales.dtype
    comptime umma_shape = Index(128 * cta_group, mma_bn, 32)

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=Index(cta_group, 1, 1),
        mma_shape=umma_shape,
        block_swizzle_size=8,
        cta_group=cta_group,
        AB_swapped=AB_swapped,
        k_group_size=1,
        num_pipeline_stages=num_pipeline_stages,
        num_accum_pipeline_stages=1 if mma_bn > 128 else 2,
        is_gmm=True,
        gemm_kind=GEMMKind.GMM,
    )

    grouped_matmul_block_scaled[transpose_b=transpose_b, config=config](
        c,
        a,
        a_offsets,
        a_scale_offsets,
        b,
        expert_ids,
        a_scales,
        b_scales,
        expert_scales,
        num_active_experts,
        ctx,
    )


def _dispatch_regime[
    transpose_b: Bool,
    N: Int,
    K: Int,
    mma_bn: Int,
    cta_group: Int,
    stages_up_proj: Optional[Int],
    stages_down_proj: Optional[Int],
](
    c: TileTensor[...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_scales: TileTensor[...],
    b_scales: TileTensor[...],
    a_offsets: TileTensor[...],
    a_scale_offsets: TileTensor[...],
    expert_ids: TileTensor[...],
    expert_scales: TileTensor[...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Dispatch with shape-specific pipeline stages.

    Uses tuned stages for known (N, K) shapes, auto-computes for others.
    """
    comptime if N == 4096 and K == 7168:
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            mma_bn,
            cta_group,
            num_pipeline_stages=stages_up_proj,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
    elif N == 7168 and K == 2048:
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            mma_bn,
            cta_group,
            num_pipeline_stages=stages_down_proj,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
    else:
        _launch_grouped_block_scaled[transpose_b, True, mma_bn, cta_group](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )


def grouped_matmul_nvfp4_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    override: Bool = False,
    AB_swapped: Bool = True,
    mma_bn: Int = 8,
    cta_group: Int = 1,
    num_pipeline_stages: Int = -1,
](
    c: TileTensor[...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_scales: TileTensor[...],
    b_scales: TileTensor[...],
    a_offsets: TileTensor[...],
    a_scale_offsets: TileTensor[...],
    expert_ids: TileTensor[...],
    expert_scales: TileTensor[...],
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
) raises:
    """Dispatch grouped NVFP4 matmul with shape-tuned configuration.

    When override=False (default, production), selects kernel parameters
    from the tuning table keyed on (N, K). The caller's
    AB_swapped/mma_bn/cta_group/num_pipeline_stages are ignored.

    When override=True (ablation/benchmarking), uses the caller's
    parameter values directly. num_pipeline_stages=-1 = auto-compute.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).
        override: If True, use caller's config params directly.
            If False, use tuning table.
        AB_swapped: A/B swap (only used when override=True).
        mma_bn: MMA tile N dimension (only used when override=True).
        cta_group: CTA group size (only used when override=True).
        num_pipeline_stages: Pipeline depth (only used when override=True).
            -1 = auto-compute.

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K//2 packed).
        b: Weight tensor B (num_experts, N, K//2 packed).
        a_scales: Scale factors for A (5D).
        b_scales: Scale factors for B (6D).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
    """
    comptime if override:
        # Ablation/benchmarking: use caller's explicit parameters.
        comptime _stages = Optional[Int](
            num_pipeline_stages
        ) if num_pipeline_stages > 0 else Optional[Int](None)
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped,
            mma_bn,
            cta_group,
            num_pipeline_stages=_stages,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
    else:
        # Production: tuning table keyed on (N, K).
        comptime N = type_of(c).static_shape[1]
        comptime packed_K = type_of(a).static_shape[1]
        comptime K = packed_K * 2  # NVFP4: 2 values per byte

        # Three regimes based on avg_m = estimated_total_m / num_active_experts:
        #   avg_m <= 8:      decode       (1SM, mma_bn=8)
        #   8 < avg_m <= 64: small prefill (2SM, mma_bn=64)
        #   avg_m > 64:      large prefill (2SM, mma_bn=128)
        if estimated_total_m <= num_active_experts * 8:
            _dispatch_regime[
                transpose_b,
                N,
                K,
                mma_bn=8,
                cta_group=1,
                stages_up_proj=6,
                stages_down_proj=4,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                num_active_experts,
                ctx,
            )
        elif estimated_total_m <= num_active_experts * 64:
            _dispatch_regime[
                transpose_b,
                N,
                K,
                mma_bn=64,
                cta_group=2,
                stages_up_proj=6,
                stages_down_proj=6,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                num_active_experts,
                ctx,
            )
        else:
            _dispatch_regime[
                transpose_b,
                N,
                K,
                mma_bn=128,
                cta_group=2,
                stages_up_proj=7,
                stages_down_proj=6,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                num_active_experts,
                ctx,
            )


def grouped_matmul_block_scaled_sm100_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
](
    c: TileTensor[...],
    a: TileTensor[...],
    b: TileTensor[...],
    a_scales: TileTensor[...],
    b_scales: TileTensor[...],
    a_offsets: TileTensor[...],
    a_scale_offsets: TileTensor[...],
    expert_ids: TileTensor[...],
    expert_scales: TileTensor[...],
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
) raises:
    """Dispatch grouped block-scaled matmul based on input dtypes.

    Routes NVFP4 to shape-tuned decode/prefill dispatch, and MXFP4/MXFP8
    to a common launcher with default configs.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K//2 packed).
        b: Weight tensor B (num_experts, N, K//2 packed).
        a_scales: Scale factors for A (5D).
        b_scales: Scale factors for B (6D).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
    """
    comptime scaling_kind = _scaling_kind[a.dtype, a_scales.dtype]()

    comptime if scaling_kind == UMMAKind.KIND_MXF4NVF4:
        grouped_matmul_nvfp4_dispatch[transpose_b, target](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
        )
    elif scaling_kind == UMMAKind.KIND_MXF4:
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped=False,
            mma_bn=128,
            cta_group=1,
            scaling_kind=UMMAKind.KIND_MXF4,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
    elif scaling_kind == UMMAKind.KIND_MXF8F6F4:
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped=False,
            mma_bn=128,
            cta_group=1,
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
