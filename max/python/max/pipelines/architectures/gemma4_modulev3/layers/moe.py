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

"""Gemma4 router and mixture of experts for the ModuleV3 API."""

from __future__ import annotations

import contextlib
import functools

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn.activation import GatedActivation
from max.nn.common_layers import quant_ops
from max.nn.common_layers.moe import MoEGate
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_moe import TensorParallelMoE
from max.nn.common_layers.quant_tensor import NVFP4Tensor, QuantAwareTensor
from max.nn.quant_config import QuantConfig
from max.nn.sequential import ModuleList
from max.tensor import Tensor, default_device, default_dtype, defaults
from max.tensor.sharding import DeviceMapping, Partial

from .rms_norm import Gemma4RMSNorm


class Gemma4Router(MoEGate):
    """Scores the experts on the normalized, learned-scaled hidden state.

    Routing runs in float32, as HF's ``Gemma4TextRouter`` does.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_dim, num_experts, num_experts_per_token)
        self.norm = Gemma4RMSNorm(hidden_dim, eps=eps, with_weight=False)
        self.scale = Tensor.ones([hidden_dim])
        self.per_expert_scale = Tensor.ones([num_experts])

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns each token's top-k experts and their weights."""
        h = self.norm(hidden_state) * self.scale * self.hidden_dim**-0.5
        logits = self.gate_score(h)
        probs = F.softmax(logits.cast(DType.float32))
        weights, idx = F.top_k(probs, k=self.num_experts_per_token, axis=-1)
        weights = weights / F.sum(weights, axis=-1)
        expert_scale = F.gather(self.per_expert_scale, idx, axis=0)
        weights = weights * expert_scale.cast(DType.float32)
        return idx, weights.cast(logits.dtype)


class Gemma4StackedExperts(ModuleList[QuantizedMLP]):
    """Declares the BF16 checkpoint's stacked expert weights under its names."""

    def __init__(self, num_experts: int, hidden_dim: int, moe_dim: int):
        super().__init__()
        self.gate_up_proj = Tensor.zeros([num_experts, 2 * moe_dim, hidden_dim])
        self.down_proj = Tensor.zeros([num_experts, hidden_dim, moe_dim])


class Gemma4MoE(TensorParallelMoE):
    """Gemma4's routed experts, which read the pre-expert-normed input.

    The router reads the raw input.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        eps: float,
        activation: GatedActivation,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=functools.partial(Gemma4Router, eps=eps),
            quant_config=quant_config,
            activation=activation,
        )
        self.pre_expert_norm = Gemma4RMSNorm(hidden_dim, eps=eps)

    def _init_experts(self) -> ModuleList[QuantizedMLP]:
        if self.quant_config is not None:
            return super()._init_experts()
        dtype, mesh = defaults()
        # As TensorParallelMoE does, keep multi-device experts on the host
        # until each device takes its slice.
        placement = (
            contextlib.nullcontext()
            if mesh.num_devices == 1
            else default_device(CPU())
        )
        with placement, default_dtype(dtype):
            return Gemma4StackedExperts(
                self.num_experts, self.hidden_dim, self.moe_dim
            )

    @property
    def gate_up_proj(self) -> list[QuantAwareTensor]:
        """Per-device ``[gate, up]`` weight bundle, sharded along ``moe_dim``."""
        experts = self.experts
        if not isinstance(experts, Gemma4StackedExperts):
            return super().gate_up_proj
        if self.mesh.num_devices == 1:
            return [experts.gate_up_proj]
        # Slice gate and up rows separately so each device's halves stay paired.
        e, h = self.num_experts, self.hidden_dim
        halves = experts.gate_up_proj.reshape([e, 2, self._initial_moe_dim, h])
        return [
            *self._shard_stack_tensors([halves], axis=2, shard_shape=[e, -1, h])
        ]

    @property
    def down_proj(self) -> list[QuantAwareTensor]:
        """Per-device down-projection bundle, sharded along ``moe_dim``.

        The fused NVFP4 gate/up weight keeps the gate's ``weight_scale_2``, so
        an expert whose up scale is ``r`` times it leaves the activation
        scaled by ``1 / r``. The down scales fold ``r`` back in.
        """
        experts = self.experts
        if isinstance(experts, Gemma4StackedExperts):
            if self.mesh.num_devices == 1:
                return [experts.down_proj]
            e, h = self.num_experts, self.hidden_dim
            return [
                *self._shard_stack_tensors(
                    [experts.down_proj], axis=2, shard_shape=[e, h, -1]
                )
            ]
        if not quant_ops.is_nvfp4_quantized(self.quant_config):
            return super().down_proj
        gate_scales: list[Tensor] = []
        up_scales: list[Tensor] = []
        downs: list[QuantAwareTensor] = []
        for expert in experts:
            gate, up = expert.gate_proj.weight, expert.up_proj.weight
            assert isinstance(gate, NVFP4Tensor)
            assert isinstance(up, NVFP4Tensor)
            gate_scales.append(gate.weight_scale_2)
            up_scales.append(up.weight_scale_2)
            downs.append(expert.down_proj.weight)
        ratio = F.stack(up_scales, axis=0) / F.stack(gate_scales, axis=0)
        stacked = quant_ops.stack(downs, axis=0)
        assert isinstance(stacked, NVFP4Tensor)
        down = NVFP4Tensor(
            weight=stacked.weight,
            weight_scale=stacked.weight_scale,
            weight_scale_2=stacked.weight_scale_2 * ratio,
            input_scale=stacked.input_scale / ratio,
            block_size=stacked.block_size,
        )
        if self.mesh.num_devices == 1:
            return [down]
        return [*down.shard(-1, self.mesh).local_shards]

    def forward(self, x: Tensor) -> Tensor:
        """Routes ``x`` and runs the experts over its pre-expert norm.

        Args:
            x: ``(seq_len, hidden_dim)``, replicated.

        Returns:
            ``(seq_len, hidden_dim)``, each device's partial sum.
        """
        router_idx, router_weight = self.gate(x)
        gate_up = quant_ops.stack_device_shards(
            self.gate_up_proj, axis=1, mesh=self.mesh
        )
        down = quant_ops.stack_device_shards(
            self.down_proj, axis=2, mesh=self.mesh
        )
        return F.call_on_mesh(
            self._routed_experts,
            self.mesh,
            out_specs=DeviceMapping(self.mesh, (Partial(),)),
        )(self.pre_expert_norm(x), router_idx, router_weight, gate_up, down)
