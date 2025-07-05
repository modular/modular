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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear
from max.pipelines.architectures.deepseekV2.layers.moe_gate import MaxMoEGate


class MoE(Module):
    num_experts_per_tok: int
    """Number of experts to router each token to."""

    ep_size: int
    """Size of expert parallel group."""

    experts_per_rank: int
    """Number of experts per rank."""

    moe_intermediate_size: int
    """Hidden dimension size for MoE intermediate layer."""

    hidden_size: int
    """Hidden dimension size for MoE layer."""

    n_shared_experts: int
    """Number of shared experts."""

    def __init__(
        self,
        device: DeviceRef,
        num_experts_per_tok: int = 6,
        ep_size: int = 1,
        experts_per_rank: int = 64,
        moe_intermediate_size: int = 1408,
        hidden_size: int = 2048,
        n_shared_experts: int = 2,
        dtype: DType = DType.bfloat16,
    ) -> None:
        """
        Args:
            device: The device the experts are on.
            num_experts_per_tok: Number of experts to route each token to.
            ep_size: Size of expert parallel group.
            experts_per_rank: Number of experts per rank.
            moe_intermediate_size: Hidden dimension size for MoE intermediate layer.
            max_position_embeddings: Maximum sequence length.
        """
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_size = ep_size
        self.experts_per_rank = experts_per_rank
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_size = hidden_size
        self.n_shared_experts = n_shared_experts
        self.gate = MaxMoEGate(device)

        # Routed experts weights
        self.down_proj = Weight(
            name="experts.down_proj.weight",
            shape=(
                self.experts_per_rank,
                self.hidden_size,
                self.moe_intermediate_size,
            ),
            dtype=dtype,
            device=device,
        )

        self.gate_proj = Weight(
            name="experts.gate_proj.weight",
            shape=(
                self.experts_per_rank,
                self.moe_intermediate_size,
                self.hidden_size,
            ),
            dtype=dtype,
            device=device,
        )

        self.up_proj = Weight(
            name="experts.up_proj.weight",
            shape=(
                self.experts_per_rank,
                self.moe_intermediate_size,
                self.hidden_size,
            ),
            dtype=dtype,
            device=device,
        )

        # Shared experts weights
        self.shared_expert_up_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.moe_intermediate_size * self.n_shared_experts,
            dtype=dtype,
            device=device,
        )
        self.shared_expert_down_proj = Linear(
            in_dim=self.moe_intermediate_size * self.n_shared_experts,
            out_dim=self.hidden_size,
            dtype=dtype,
            device=device,
        )
        self.shared_expert_gate_proj = Linear(
            in_dim=self.hidden_size,
            out_dim=self.moe_intermediate_size * self.n_shared_experts,
            dtype=dtype,
            device=device,
        )

    def __call__(self, hidden_states: TensorValue):
        """Mixture of Experts Layer.

        Args:
            hidden_states: Input tensor of shape (seq_length, hidden_size)

        Returns:
            Output tensor of shape (seq_length, hidden_size)
        """

        identity = hidden_states
        # Get the topk experts per token and their weights
        topk_idx, topk_weight = self.gate(hidden_states)

        # Gather the weights for the topk experts for each token
        # (seq_len, k, h, w)
        topk_down_proj = ops.gather(self.down_proj, topk_idx, axis=0)
        topk_gate_proj = ops.gather(self.gate_proj, topk_idx, axis=0)
        topk_up_proj = ops.gather(self.up_proj, topk_idx, axis=0)

        # Unsqueeze the hidden states to match the shape of the topk weights
        # (seq_len, w) -> (seq_len, w, 1)
        hidden_states = ops.unsqueeze(
            ops.unsqueeze(hidden_states, axis=1), axis=-1
        )

        # (seq_len, k, h, w) @ (seq_len, 1, w, 1) -> (seq_len, k, h, 1)
        up_projs = topk_up_proj @ hidden_states

        # (seq_len, k, h, w) @ (seq_len, 1, w, 1) -> (seq_len, k, h, 1)
        gate_projs = topk_gate_proj @ hidden_states

        # apply silu to gate_projs.
        gate_projs = ops.silu(gate_projs)

        # (seq_len, k, h, 1) * (seq_len, k, h, 1) -> (seq_len, k, h, 1)
        up_gate_projs = up_projs * gate_projs

        # (seq_len, k, w, h) @ (seq_len, k, h, 1) -> (seq_len, k, w)
        down_projs = ops.squeeze(topk_down_proj @ up_gate_projs, axis=-1).cast(
            topk_weight.dtype
        )
        topk_weight = ops.unsqueeze(topk_weight, axis=1)

        # (seq_len, 1, k) @ (seq_len, k, w) -> (seq_len, 1, w)
        summed_down_projs = (topk_weight @ down_projs).cast(identity.dtype)
        final_out = ops.squeeze(summed_down_projs, axis=1)

        # TODO(MODELS-396): Probably should be a MLP layer
        shared_expert_out = self.shared_expert_down_proj(
            ops.silu(self.shared_expert_gate_proj(identity))
            * self.shared_expert_up_proj(identity)
        )

        return final_out + shared_expert_out
