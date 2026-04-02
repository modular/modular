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
"""Gemma4 Mixture of Experts."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.layer import Module
from max.nn.linear import Linear
from max.pipelines.architectures.diancie.layers.rms_norm import Gemma4RMSNorm


class Gemma4TextRouter(Module):
    """Gemma4 MoE router that selects top-k experts per token."""

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_experts_per_token = num_experts_per_token
        self.scalar_root_size = hidden_dim**-0.5

        self.norm = Gemma4RMSNorm(
            dim=hidden_dim, dtype=dtype, eps=eps, with_weight=False
        )
        self.scale = Weight(
            name="scale", dtype=dtype, shape=[hidden_dim], device=device
        )
        self.proj = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.per_expert_scale = Weight(
            name="per_expert_scale",
            dtype=dtype,
            shape=[num_experts],
            device=device,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Routes tokens to top-k experts.

        Args:
            hidden_states: Input tensor of shape ``[seq_len, hidden_dim]``.

        Returns:
            A tuple of ``(top_k_weights, top_k_index)`` each of shape
            ``[seq_len, num_experts_per_token]``.
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probs = ops.softmax(expert_scores)

        top_k_weights, top_k_index = ops.top_k(
            router_probs, k=self.num_experts_per_token, axis=-1
        )

        # Renormalize top-k weights to sum to 1 per token.
        top_k_weights = top_k_weights / ops.sum(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * ops.gather(
            self.per_expert_scale, top_k_index, axis=0
        )

        return top_k_weights, top_k_index


class Gemma4TextExperts(Module):
    """Gemma4 Mixture of Experts block with per-expert scaling.

    Unlike the standard ``StackedMoE``, this block does not contain a
    gate/router - ``top_k_index`` and ``top_k_weights`` are passed as
    inputs from the caller.

    Weight shapes match the HuggingFace convention:

    - ``gate_up_proj``: ``[num_experts, 2 * intermediate_dim, hidden_dim]``
    - ``down_proj``: ``[num_experts, hidden_dim, intermediate_dim]``
    - ``per_expert_scale``: ``[num_experts]``
    """

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
        num_experts: int,
        num_experts_per_token: int,
        hidden_dim: int,
        intermediate_dim: int,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        self.gate_up_proj = Weight(
            name="gate_up_proj",
            dtype=dtype,
            shape=[num_experts, 2 * intermediate_dim, hidden_dim],
            device=device,
        )
        self.down_proj = Weight(
            name="down_proj",
            dtype=dtype,
            shape=[num_experts, hidden_dim, intermediate_dim],
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        top_k_index: TensorValue,
        top_k_weights: TensorValue,
    ) -> TensorValue:
        """Applies MoE expert computation.

        Args:
            hidden_states: Input tensor of shape ``[seq_len, hidden_dim]``.
            top_k_index: Expert indices of shape ``[seq_len, k]``.
            top_k_weights: Routing weights of shape ``[seq_len, k]``.

        Returns:
            Output tensor of shape ``[seq_len, hidden_dim]``.
        """
        seq_len = hidden_states.shape[0]

        router_idx_flat = ops.reshape(top_k_index, [-1])

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx_flat, DType.int32), self.num_experts
        )

        token_indices = ops.cast(
            token_expert_order // self.num_experts_per_token, DType.int32
        )
        permuted_states = ops.gather(hidden_states, token_indices, axis=0)

        gate_up_output = grouped_matmul_ragged(
            permuted_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        gate = gate_up_output[:, : self.intermediate_dim]
        up = gate_up_output[:, self.intermediate_dim :]
        gated_output = ops.gelu(gate, approximate="tanh") * up

        down_output = grouped_matmul_ragged(
            gated_output,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        down_output = ops.gather(
            down_output, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, self.hidden_dim])

        routed_expert_out = ops.unsqueeze(top_k_weights, axis=1) @ down_output
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(
            hidden_states.dtype
        )

        return routed_expert_out
