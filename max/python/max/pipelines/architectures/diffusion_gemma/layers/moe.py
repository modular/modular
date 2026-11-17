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
"""Mixture of experts for DiffusionGemma.

The gate routes each token as HF's ``DiffusionGemmaTextRouter`` does, with
the router softmax in float32. The BF16 checkpoints stack each projection's
experts into one tensor; the NVFP4 checkpoints store each expert on its own.
"""

from __future__ import annotations

import functools

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.activation import GatedActivation
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.stacked_quant_moe import StackedQuantizedMoE
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor

from .rms_norm import Gemma4RMSNorm

# HF ``DiffusionGemmaTextMLP`` uses ``gelu_pytorch_tanh``.
_GELU_TANH = GatedActivation("gelu_tanh")


class DiffusionGemmaMoEGate(MoEGate):
    """Gemma4 MoE gate with RMSNorm, learned scale, and per-expert scaling.

    The router softmax, top-k renormalization, and the per-expert scale are
    applied in float32 (HF ``DiffusionGemmaTextRouter.forward``), and the
    weights are cast back to the score dtype for the expert combine.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.scalar_root_size = hidden_dim**-0.5
        # The norm has no weights, so it is the same on every device.
        self.norm = Gemma4RMSNorm(dim=hidden_dim, eps=eps, with_weight=False)
        self.scale = Tensor.zeros([hidden_dim])
        self.per_expert_scale = Tensor.zeros([num_experts])

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each token's top-k expert indices and weights."""
        hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state * self.scale * self.scalar_root_size

        expert_scores = self.gate_score(hidden_state)
        router_probs = F.softmax(F.cast(expert_scores, DType.float32))

        top_k_weights, top_k_index = F.top_k(
            router_probs, k=self.num_experts_per_token, axis=-1
        )
        top_k_weights = top_k_weights / F.sum(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * F.cast(
            F.gather(self.per_expert_scale, top_k_index, axis=0),
            DType.float32,
        )
        return top_k_index, F.cast(top_k_weights, expert_scores.dtype)


def _routed_forward(
    moe: QuantizedMoE, pre_expert_norm: Gemma4RMSNorm, x: Tensor
) -> Tensor:
    # The gate scores the raw residual; only the experts see the normed one.
    router_idx, router_weight = moe.gate(x)
    x = pre_expert_norm(x)
    return F.call_on_mesh(
        moe._routed_experts, x.mesh, out_specs=router_weight.mapping
    )(x, router_idx, router_weight, moe.gate_up_proj[0], moe.down_proj[0])


class DiffusionGemmaMoE(StackedQuantizedMoE):
    """DiffusionGemma's routed experts over the BF16 stacked weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        eps: float,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=functools.partial(DiffusionGemmaMoEGate, eps=eps),
            activation=_GELU_TANH,
        )
        self.pre_expert_norm = Gemma4RMSNorm(dim=hidden_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Routes ``x`` through its top-k experts and combines the outputs."""
        return _routed_forward(self, self.pre_expert_norm, x)


class DiffusionGemmaQuantizedMoE(QuantizedMoE):
    """DiffusionGemma's routed experts over NVFP4 per-expert weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        eps: float,
        quant_config: QuantConfig,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=functools.partial(DiffusionGemmaMoEGate, eps=eps),
            quant_config=quant_config,
            activation=_GELU_TANH,
        )
        self.pre_expert_norm = Gemma4RMSNorm(dim=hidden_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Routes ``x`` through its top-k experts and combines the outputs."""
        return _routed_forward(self, self.pre_expert_norm, x)


def build_diffusion_gemma_moe(
    *,
    hidden_dim: int,
    num_experts: int,
    num_experts_per_token: int,
    moe_dim: int,
    eps: float,
    quant_config: QuantConfig | None,
) -> Module[[Tensor], Tensor]:
    """Builds the routed-expert block a checkpoint's encoding calls for."""
    if quant_config is not None and quant_config.is_nvfp4:
        return DiffusionGemmaQuantizedMoE(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            eps=eps,
            quant_config=quant_config,
        )
    return DiffusionGemmaMoE(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        moe_dim=moe_dim,
        eps=eps,
    )


__all__ = [
    "DiffusionGemmaMoE",
    "DiffusionGemmaMoEGate",
    "DiffusionGemmaQuantizedMoE",
    "build_diffusion_gemma_moe",
]
