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
from max.graph import TensorValue, Weight
from max.nn.layer import Module
from max.nn.linear import LinearV2


class MoE(Module):
    hidden_dim: int
    """Hidden dimension size."""

    top_k: int
    """Number of experts to route each token to."""

    num_experts: int
    """Total number of experts."""

    dtype: DType
    """Data type for weights."""

    def __init__(
        self,
        hidden_dim: int = 5120,
        top_k: int = 1,
        num_experts: int = 16,
        dtype: DType = DType.bfloat16,
    ):
        """
        Args:
            num_experts_per_tok: Number of experts to route each token to.
            ep_size: Size of expert parallel group.
            experts_per_rank: Number of experts per rank.
            moe_intermediate_size: Hidden dimension size for MoE intermediate layer.
            max_position_embeddings: Maximum sequence length.
        """
        super().__init__()

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # TODO: Weight shapes are hardcoded for now. Will be updated to use parameters from config.

        # Routed experts weights
        self.down_proj = Weight(
            name="experts.down_proj.weight",
            shape=(
                16,
                8192,
                5120,
            ),
            dtype=dtype,
        )

        self.gate_up_proj = Weight(
            name="experts.gate_up_proj.weight",
            shape=(
                16,
                5120,
                16384,
            ),
            dtype=dtype,
        )

        # Shared experts weights
        self.shared_expert_up_proj = LinearV2(
            in_dim=5120,
            out_dim=8192,
            dtype=dtype,
        )
        self.shared_expert_down_proj = LinearV2(
            in_dim=8192,
            out_dim=5120,
            dtype=dtype,
        )
        self.shared_expert_gate_proj = LinearV2(
            in_dim=5120,
            out_dim=8192,
            dtype=dtype,
        )

        self.router = LinearV2(
            in_dim=self.hidden_dim,
            out_dim=self.num_experts,
            dtype=dtype,
        )

    def __call__(self, hidden_states: TensorValue):
        raise NotImplementedError("Llama4 MoE layer not yet implemented")

        return None
