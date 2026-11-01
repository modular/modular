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
(``group_limited_router_kernel``) both dispatch to. This subclasses
:class:`DeepseekV3TopKRouter` rather than reimplementing the scoring.

:class:`DeepseekV3TopKRouter` only casts the gate's *output* to
``correction_bias_dtype`` after the matmul; it does not force
``gate_score``'s own weight to fp32, so the matmul itself would run in a
lower precision than the reference, which casts both the activation and the
weight to fp32 *before* the matmul. Constructing the router inside a fp32
``default_dtype`` context fixes this: ``gate_score`` is a plain
:class:`~max.nn.Linear`, built from whatever the ambient
default dtype is at construction time, so it becomes fp32 too.

On gfx950, a bf16 activation on an accelerator device takes the fused mixed
router GEMV (``mo.router.gate.mixed.gemv``): the activation is widened to
fp32 in registers and dotted against the unchanged fp32 gate weight in one
launch, replacing the inherited path's tiled fp32 GEMM, which at the small-M
decode row count pays a full gate-weight read to produce a handful of rows.
Both compute fp32 arithmetic over exactly the same widened values, in
different summation orders, so results can round differently. Every other
device, activation dtype and gate shape keeps the inherited path.
"""

from __future__ import annotations

from max import functional as F
from max.driver import accelerator_architecture_name, accelerator_count
from max.dtype import DType
from max.graph import StaticDim
from max.nn.common_layers.functional_kernels import moe_router_group_limited
from max.nn.legacy.kernels import _router_gate_mixed_gemv
from max.pipelines.architectures.deepseekV3_modulev3.layers.moe_gate import (
    DeepseekV3TopKRouter,
)
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import match_operand_placement

# The output takes the activation's placement: under TP the router stays
# replicated, so the logits the fused GEMV returns are replicated too.
_router_gate_mixed_gemv_op = F.functional(
    _router_gate_mixed_gemv,
    rule=match_operand_placement(_router_gate_mixed_gemv, "hidden_states"),
)

# K granule the fused GEMV walks the gate weight's contraction by on gfx950:
# `simd_width * num_threads` -- 512 for the 128-thread launches every gate
# shape but [896, 7168] takes, 1024 on that table's streaming and wide-K
# bands (max/kernels/src/linalg/gemv.mojo). The K walk has no tail guard, so
# the op is only correct for a K that tiles its band's granule; [896, 7168]
# pins K at 7168, which tiles every band, and every other shape walks the
# 512 granule, so checking 512 is exact. Kimi K3's 7168 qualifies, as does
# the other caller's 6144.
_MIXED_GEMV_K_TILE = 512


def _mixed_gemv_active(hidden_states: Tensor) -> bool:
    """Returns whether this router call can run the fused mixed router GEMV.

    Two conditions, both required. The graph's own device: the op is
    registered GPU-only, so a CPU-device graph keeps the inherited path even
    on a gfx950 host. And the host accelerator query this package's AMD W4A8
    validation reads (``amd_w4a8_validation.validate_amd_tp_w4a8``), pinned
    to the one part the op's AMD launch table was tuned for: K3 ships no
    NVIDIA recipe, and the op's SM100 table was tuned for a different gate
    shape.

    Reads the placement off ``hidden_states.mesh.devices`` rather than
    ``Tensor.device``, which raises on the multi-device meshes a
    tensor-parallel router input lives on. The architecture query raises on
    a host with no accelerator, so the count check guards it.
    """
    return (
        not all(device.is_host for device in hidden_states.mesh.devices)
        and accelerator_count() > 0
        and accelerator_architecture_name() == "gfx950"
    )


def _gate_weight_tiles_k(weight: Tensor) -> bool:
    """Returns whether the fused GEMV can walk ``weight``'s K in whole tiles.

    A statically known misaligned K would read past the weight rows, so
    those shapes keep the inherited cast + matmul path. A dynamic K cannot
    be checked here; it takes the fused path, whose wrapper then fails fast
    on it.
    """
    k_dim = weight.shape[1]
    return not isinstance(k_dim, StaticDim) or (
        int(k_dim) % _MIXED_GEMV_K_TILE == 0
    )


class KimiK3TopKRouter(DeepseekV3TopKRouter):
    """K3's ``noaux_tc`` router, fusing the gate GEMV where it applies.

    Everything past the gate logits -- sigmoid, correction bias,
    group-limited top-k -- is the inherited scoring, restated so the fused
    GEMV can replace only the projection that produces the logits.
    """

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Computes expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape ``(seq_len, hidden_dim)``.

        Returns:
            A pair ``(topk_idx, topk_weight)`` of selected expert indices and
            their routing weights, each of shape
            ``(seq_len, num_experts_per_token)``.
        """
        if not (
            _mixed_gemv_active(hidden_states)
            and hidden_states.dtype == DType.bfloat16
            and _gate_weight_tiles_k(self.gate_score.weight)
        ):
            return super().forward(hidden_states)

        # Unconditional on this path: a gate weight the fused op cannot bind
        # (a non-fp32 dtype, a dynamic shape) raises at graph build rather
        # than silently taking the slower inherited path.
        logits = _router_gate_mixed_gemv_op(
            hidden_states, self.gate_score.weight
        )
        scores = F.sigmoid(logits.cast(self.e_score_correction_bias.dtype))
        topk_idx, topk_weight = moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
        return (
            topk_idx.rebind_mapping(scores.mapping),
            topk_weight.rebind_mapping(scores.mapping),
        )


def make_kimi_k3_moe_router(
    hidden_dim: int,
    num_experts: int,
    num_experts_per_token: int,
    routed_scaling_factor: float,
    num_expert_group: int,
    topk_group: int,
    moe_renormalize: bool,
) -> KimiK3TopKRouter:
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
        return KimiK3TopKRouter(
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
