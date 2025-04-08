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
from max.graph import TensorValue, Weight, ops
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
        """
        super().__init__()

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.intermediate_size_mlp = intermediate_size_mlp

        # Routed experts weights
        self.down_proj = Weight(
            name="experts.down_proj.weight",
            shape=(
                self.num_experts,
                self.intermediate_size,
                self.hidden_dim,
            ),
            dtype=dtype,
        )

        self.gate_up_proj = Weight(
            name="experts.gate_up_proj.weight",
            shape=(
                self.num_experts,
                self.hidden_dim,
                self.intermediate_size_mlp,
            ),
            dtype=dtype,
        )

        # Shared experts weights
        self.shared_expert_up_proj = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
        )
        self.shared_expert_down_proj = LinearV2(
            in_dim=self.intermediate_size,
            out_dim=self.hidden_dim,
            dtype=dtype,
        )
        self.shared_expert_gate_proj = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.intermediate_size,
            dtype=dtype,
        )

        self.router = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.num_experts,
            dtype=dtype,
        )

    def __call__(self, hidden_states: TensorValue):
        hidden_states = ops.reshape(hidden_states, (-1, self.hidden_dim))
        router_logits = self.router(hidden_states)
        # (batch * seq_len, num_experts)
        top_idx = ops.squeeze(ops.argmax(router_logits, axis=-1), axis=1)
        # (batch * seq_len,)
        router_probs = ops.sigmoid(ops.max(router_logits, axis=-1))
        # (batch * seq_len, 1)

        top_down_proj = ops.gather(self.down_proj, top_idx, axis=0)
        top_gate_up_proj = ops.gather(self.gate_up_proj, top_idx, axis=0)
        # (batch * seq_len, h, w)

        gate_up_projs = ops.unsqueeze(hidden_states, axis=1) @ top_gate_up_proj
        # (batch * seq_len, 1, hidden_dim) @ (batch*seq_len, hidden_dim, intermediate_size) -> (batch*seq_len, 1, intermediate_size)

        gate_up_projs = (
            ops.silu(
                gate_up_projs[:, :, : self.intermediate_size].cast(
                    DType.float32
                )
            ).cast(DType.bfloat16)
            * gate_up_projs[:, :, self.intermediate_size :]
        )

        down_projs = ops.squeeze(gate_up_projs @ top_down_proj, axis=1)
        # (batch * seq_len, 16, 8192) @ (batch*seq_len, intermediate_size, 1) -> (batch * seq_len, hidden_dim)

        shared_expert_out = self.shared_expert_down_proj(
            ops.silu(
                self.shared_expert_gate_proj(hidden_states).cast(DType.float32)
            ).cast(DType.bfloat16)
            * self.shared_expert_up_proj(hidden_states)
        )
        return router_probs * down_projs + shared_expert_out
