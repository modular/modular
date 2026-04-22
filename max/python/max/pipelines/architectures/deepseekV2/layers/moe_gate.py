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

"""Mixture of Experts Gate Layer."""

from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorValue, ops
from max.nn.moe import MoEGate


def _fill(
    fill_value: bool, dtype: DType, shape: Shape, device: DeviceRef
) -> TensorValue:
    return ops.constant(fill_value, dtype=dtype, device=device).broadcast_to(
        shape
    )


class DeepSeekV2MoEGate(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V2."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        topk_method: str,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: DType,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoEGate.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            topk_method: The method to select the top-k experts. Supported
                methods: "greedy", "group_limited_greedy"
            n_group: The number of groups (with group_limited_greedy)
            topk_group: The number of top k groups (with group_limited_greedy)
            routed_scaling_factor: The scaling factor for the routed experts
                when using group_limited_greedy top-k implementation.
            dtype: The data type of the MoEGate.
        """
        super().__init__(
            devices, hidden_dim, num_experts, num_experts_per_token, dtype
        )

        if topk_method not in ["greedy", "group_limited_greedy"]:
            raise ValueError(f"Invalid topk_method: {topk_method}")

        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        # compute gating score
        logits = self.gate_score(hidden_states.cast(DType.float32))
        scores = ops.softmax(logits.cast(DType.float32))

        # select top k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = ops.top_k(
                scores, self.num_experts_per_token, -1
            )
        elif self.topk_method == "group_limited_greedy":
            bsz_seq_len, _ = hidden_states.shape
            group_scores = ops.max(
                scores.reshape((bsz_seq_len, self.n_group, -1)), axis=-1
            )
            group_scores = ops.squeeze(group_scores, -1)  # [n, n_group]

            # Shape of group_idx: [n, top_k_group]
            group_idx = ops.top_k(group_scores, k=self.topk_group, axis=-1)[1]

            group_mask = _fill(
                False, DType.bool, group_scores.shape, group_scores.device
            )  # [n, n_group]
            update = _fill(
                True, DType.bool, group_idx.shape, group_scores.device
            )
            group_mask = ops.scatter(group_mask, update, group_idx, 1)

            score_mask = ops.broadcast_to(
                ops.unsqueeze(group_mask, -1),
                (
                    bsz_seq_len,
                    self.n_group,
                    self.num_experts // self.n_group,
                ),
            ).reshape((bsz_seq_len, -1))  # [n, e]

            tmp_scores = ops.where(
                score_mask.cast(DType.bool),
                scores,
                ops.constant(0, dtype=scores.dtype, device=scores.device),
            )  # [n, e]

            topk_weight, topk_idx = ops.top_k(
                tmp_scores, k=self.num_experts_per_token, axis=-1
            )
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight
