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
"""Kimi K3's ``noaux_tc`` sigmoid MoE router (ModuleV3).

The scoring math -- biased top-k selection, unbiased weight gather, group
scoring by the sum of each group's top-2 biased scores, fp32 gate -- matches
DeepSeek V3's ``noaux_tc`` router exactly, down to the shared Mojo kernel
(``group_limited_router_kernel``) both dispatch to. This reuses
:class:`DeepseekV3TopKRouter` rather than reimplementing the scoring.

:class:`DeepseekV3TopKRouter` only casts the gate's *output* to
``correction_bias_dtype`` after the matmul; it does not force
``gate_score``'s own weight to fp32, so the matmul itself would run in a
lower precision than the reference, which casts both the activation and the
weight to fp32 *before* the matmul. Constructing the router inside a fp32
``default_dtype`` context fixes this: ``gate_score`` is a plain
:class:`~max.nn.Linear`, built from whatever the ambient
default dtype is at construction time, so it becomes fp32 too.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental.tensor import default_dtype
from max.pipelines.architectures.deepseekV3_modulev3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)


def make_kimi_k3_moe_router(
    hidden_dim: int,
    num_experts: int,
    num_experts_per_token: int,
    routed_scaling_factor: float,
    num_expert_group: int,
    topk_group: int,
    moe_renormalize: bool,
) -> DeepseekV3TopKRouter:
    """Builds K3's MoE router, with the gate matmul forced to run in fp32.

    Args:
        hidden_dim: ``config.hidden_size`` -- the gate reads the FULL
            pre-latent hidden state, not the routed experts' narrower latent
            width.
        num_experts: ``config.num_experts``.
        num_experts_per_token: ``config.num_experts_per_token``.
        routed_scaling_factor: ``config.routed_scaling_factor``.
        num_expert_group: ``config.num_expert_group``.
        topk_group: ``config.topk_group``.
        moe_renormalize: ``config.moe_renormalize``.
    """
    with default_dtype(DType.float32):
        return DeepseekV3TopKRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=num_expert_group,
            topk_group=topk_group,
            norm_topk_prob=moe_renormalize,
            correction_bias_dtype=DType.float32,
        )
