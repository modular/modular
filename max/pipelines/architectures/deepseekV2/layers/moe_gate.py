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

"""Mixture of Experts Gate Layer."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Linear
from max.nn.layer import Module


class MaxMoEGate(Module):
    """Mixture of Experts Gate Layer."""

    num_experts_per_tok: int
    """Number of experts to router each token to."""

    n_routed_experts: int
    """Total number of experts in the model."""

    routed_scaling_factor: float
    """Scaling factor for routing weights."""

    aux_loss_alpha: float
    """Weight for auxiliary loss."""

    n_group: int
    """Number of groups for expert routing."""

    topk_group: int
    """Number of top experts per group."""

    gating_dim: int
    """Hidden dimension size for gating."""

    def __init__(
        self,
        device: DeviceRef,
        num_experts_per_tok: int = 6,
        n_routed_experts: int = 64,
        routed_scaling_factor: float = 1.0,
        aux_loss_alpha: float = 0.001,
        n_group: int = 1,
        topk_group: int = 1,
        gating_dim: int = 2048,  # equal to config.hidden_size
    ) -> None:
        """
        Args:
            device: The device this layer's weights are on.
            gate_score: Linear layer that projects from hidden_size to intermediate_size.
            num_experts_per_tok: Number of experts to route each token to.
            n_routed_experts: Total number of experts in the model.
            routed_scaling_factor: Scaling factor for routing weights.
            aux_loss_alpha: Weight for auxiliary loss.
            n_group: Number of groups for expert routing.
            topk_group: Number of top experts per group.
            gating_dim: Hidden dimension size for gating.
        Shape:
            Input: (batch_size, seq_length, hidden_size)
            Output: tuple of:
                - topk_idx: (batch_size * seq_length, num_experts_per_tok)
                - topk_weight: (batch_size * seq_length, num_experts_per_tok)
        """
        super().__init__()
        self.gate_score = Linear(
            in_dim=gating_dim,
            out_dim=n_routed_experts,
            dtype=DType.bfloat16,
            device=device,
            has_bias=False,
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.aux_loss_alpha = aux_loss_alpha
        self.n_group = n_group
        self.topk_group = topk_group
        self.gating_dim = gating_dim

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (batch_size * seq_length, num_experts_per_tok)
                - topk_weight: Routing weights for selected experts of shape (batch_size * seq_length, num_experts_per_tok)
        """

        # compute gating score
        logits = self.gate_score(hidden_states.cast(DType.float32))
        scores = ops.softmax(logits.cast(DType.float32))

        # select top k experts
        topk_weight, topk_idx = ops.top_k(scores, self.num_experts_per_tok, -1)

        return topk_idx, topk_weight
