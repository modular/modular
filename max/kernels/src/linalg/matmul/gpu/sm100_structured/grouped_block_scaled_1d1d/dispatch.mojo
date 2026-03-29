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
"""Dispatch logic for grouped 1D-1D block-scaled SM100 NVFP4 matmul.

Selects optimal kernel configuration based on (N, K) shape and workload
type (decode vs prefill), with parameters tuned via ablation on B200.

When `override=True`, uses the caller's AB_swapped/mma_bn/cta_group/
num_pipeline_stages directly (for ablation studies and benchmarking).
When `override=False` (default), ignores those parameters and selects
from the tuning table based on (N, K) and `estimated_total_m`.

Tuning table (keyed on N, K):

  Decode (estimated_total_m < num_active_experts * 8):
  - N=4096, K=7168:  AB_swapped=True, mma_bn=8, cta_group=1, stages=6
  - N=7168, K=2048:  AB_swapped=True, mma_bn=8, cta_group=1, stages=4
  - Default:         AB_swapped=True, mma_bn=8, cta_group=1, stages=auto

  Prefill (estimated_total_m >= num_active_experts * 8):
  - N=4096, K=7168:  AB_swapped=True, mma_bn=128, cta_group=2, stages=7
  - N=7168, K=2048:  AB_swapped=True, mma_bn=128, cta_group=2, stages=6
  - Default:         AB_swapped=True, mma_bn=128, cta_group=2, stages=auto
"""

from std.collections import Optional

from std.gpu.host import DeviceContext
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.utils.index import Index
from layout import TileTensor

from ..structured_kernels.config import BlockScaledMatmulConfig
from .grouped_1d1d_matmul import grouped_matmul_block_scaled


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
        scaling_kind: Block-scaling format (only NVFP4 currently supported).
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


def _dispatch_decode[
    transpose_b: Bool, N: Int, K: Int
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
    """Decode tuning table: small M, AB_swapped=True, mma_bn=8."""
    comptime if N == 4096 and K == 7168:
        # DeepSeek V3/Kimi K2.5 up-projection: stages=6
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            8,
            1,
            num_pipeline_stages=6,
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
        # DeepSeek V3/Kimi K2.5 down-projection: stages=4
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            8,
            1,
            num_pipeline_stages=4,
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
        # Default decode: auto-compute pipeline stages
        _launch_grouped_block_scaled[transpose_b, True, 8, 1](
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


def _dispatch_prefill[
    transpose_b: Bool, N: Int, K: Int
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
    """Prefill tuning table: large M, AB_swapped=True, mma_bn=128, cta_group=2.

    Tuned pipeline stages from ablation on B200:
      N=4096, K=7168 (up-proj):  stages=7
      N=7168, K=2048 (down-proj): stages=6
    """
    comptime if N == 4096 and K == 7168:
        # DeepSeek V3/Kimi K2.5 up-projection: stages=7
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            128,
            2,
            num_pipeline_stages=7,
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
        # DeepSeek V3/Kimi K2.5 down-projection: stages=6
        _launch_grouped_block_scaled[
            transpose_b,
            True,
            128,
            2,
            num_pipeline_stages=6,
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
        # Default prefill: auto-compute pipeline stages
        _launch_grouped_block_scaled[transpose_b, True, 128, 2](
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

        # The prefill optimized kernel is faster when on average there are more
        # than 8 tokens per expert.
        if estimated_total_m > num_active_experts * 8:
            _dispatch_prefill[transpose_b, N, K](
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
            _dispatch_decode[transpose_b, N, K](
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
