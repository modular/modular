# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Mixture of Experts Layer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.layer import Module
from max.nn.linear import LinearV2


class MoE(Module):
    hidden_dim: int
    """Hidden dimension size."""

    top_k: int
    """Number of experts to route each token to."""

    num_experts: int
    """Total number of experts."""

    intermediate_size: int
    """Hidden dimension size for MoE intermediate layer."""

    intermediate_size_mlp: int
    """Hidden dimension size for MoE experts mlp layer."""

    dtype: DType
    """Data type for weights."""

    def __init__(
        self,
        device: DeviceRef,
        hidden_dim: int = 5120,
        top_k: int = 1,
        num_experts: int = 16,
        intermediate_size: int = 8192,
        intermediate_size_mlp: int = 16384,
        dtype: DType = DType.bfloat16,
    ):
        """
        Args:
            hidden_dim: Hidden dimension size.
            top_k: Number of experts to route each token to.
            num_experts: Total number of experts.
            intermediate_size: Hidden dimension size for MoE intermediate layer.
            intermediate_size_mlp: Hidden dimension size for MoE MLP layer.
            dtype: Data type for weights.
            device: The device to use to run this layer.
        """
        super().__init__()
        if not device:
            raise ValueError("Device must be provided.")

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp
        self.device = device

        if self.top_k > 1:
            raise NotImplementedError(
                "Multiple expert routing (top-k > 1) is not yet implemented. "
                "This layer currently only supports single expert routing (top-k = 1) "
                "as used in the Llama-4-Scout-17B-16E-Instruct model."
            )

        # Routed experts weights.
        # These weights are read on CPU and then are explicitly transferred to
        # GPU.
        self.down_proj = Weight(
            name="experts.down_proj",
            shape=(
                self.num_experts,
                self.intermediate_size,
                self.hidden_dim,
            ),
            dtype=dtype,
            device=device,
        )

        self.gate_up_proj = Weight(
            name="experts.gate_up_proj",
            shape=(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_size_mlp,
            ),
            dtype=dtype,
            device=device,
        )

        # Shared experts weights
        self.shared_expert_up_proj = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
            device=device,
        )
        self.shared_expert_down_proj = LinearV2(
            in_dim=self.intermediate_size,
            out_dim=self.hidden_dim,
            dtype=dtype,
            device=device,
        )
        self.shared_expert_gate_proj = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
            device=device,
        )

        self.router = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.num_experts,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        **unused_kwargs,
    ) -> list[TensorValue]:
        hidden_states = hidden_states[0]
        assert hidden_states.device == self.device
        hidden_states = ops.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = self.router(hidden_states)
        # (batch * seq_len, num_experts)
        top_idx = ops.squeeze(ops.argmax(router_logits, axis=-1), axis=1)
        # (batch * seq_len,)
        router_probs = ops.sigmoid(
            ops.max(router_logits, axis=-1).cast(DType.float32)
        ).cast(hidden_states.dtype)
        # (batch * seq_len, 1)

        down_proj_weight = self.down_proj.to(self.device)
        gate_up_proj_weight = self.gate_up_proj.to(self.device)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(top_idx, DType.uint32),
            self.num_experts,
        )

        permutated_states = ops.gather(
            router_probs, token_expert_order, axis=0
        ) * ops.gather(hidden_states, token_expert_order, axis=0)
        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            ops.transpose(gate_up_proj_weight, 1, 2),
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.intermediate_size])
            * gate_up_projs[:, self.intermediate_size :]
        )

        down_projs = grouped_matmul_ragged(
            gate_up_projs,
            ops.transpose(down_proj_weight, 1, 2),
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        )

        routed_expert_out = ops.gather(down_projs, restore_token_order, axis=0)

        shared_expert_out = self.shared_expert_down_proj(
            ops.silu(self.shared_expert_gate_proj(hidden_states))
            * self.shared_expert_up_proj(hidden_states)
        )
        return [routed_expert_out + shared_expert_out]
