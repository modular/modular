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

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorValue, Weight, ops
from max.nn.moe import MoEGate
from max.nn.moe.moe import ShardingStrategy


def _fill(
    fill_value: bool, dtype: DType, shape: Shape, device: DeviceRef
) -> TensorValue:
    return ops.constant(fill_value, dtype=dtype, device=device).broadcast_to(
        shape
    )


class DeepseekV3TopKRouter(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V3."""

    def __init__(
        self,
        num_experts_per_token: int,
        num_experts: int,
        routed_scaling_factor: float,
        scoring_func: str,
        topk_method: str,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        hidden_dim: int,
        dtype: DType,
        gate_dtype: DType,
        correction_bias_dtype: DType | None,
        devices: list[DeviceRef],
    ) -> None:
        """
        Args:
            num_experts_per_token: The number of experts per token.
            num_experts: The number of experts to route to.
            routed_scaling_factor: The scaling factor for the routed experts.
            scoring_func: The scoring function for the experts.
            topk_method: The method to select the top-k experts.
            n_group: The number of groups.
            topk_group: The number of top-k groups.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            hidden_dim: The dimension of the hidden state.
            dtype: The data type of the MoEGate.
            correction_bias_dtype: The data type of the correction bias.
            devices: The devices to use for the MoEGate.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=gate_dtype,
        )

        if topk_method not in ["noaux_tc"]:
            raise ValueError(f"Invalid topk_method: {topk_method}")

        # This value is renamed to top_k in the original implementation, keep it
        # here for consistency.
        self.top_k = num_experts_per_token

        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.gate_dtype = gate_dtype
        self.correction_bias_dtype = correction_bias_dtype

        if self.topk_method == "noaux_tc":
            if correction_bias_dtype is None:
                raise ValueError(
                    "correction_bias_dtype is required for topk_method=noaux_tc"
                )
            self.e_score_correction_bias = Weight(
                "e_score_correction_bias",
                shape=[self.num_experts],
                device=self.devices[0],
                dtype=correction_bias_dtype,
            )

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
        bsz_seq_len, _ = hidden_states.shape

        # compute gate score
        weight = self.gate_score.weight.cast(DType.float32).to(
            hidden_states.device
        )
        logits = hidden_states.cast(DType.float32) @ weight.T

        if self.scoring_func == "sigmoid":
            scores = ops.sigmoid(logits)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # select top-k experts
        if self.topk_method == "noaux_tc":
            scores_for_choice = scores.reshape(
                (bsz_seq_len, -1)
            ) + ops.unsqueeze(self.e_score_correction_bias, 0)
            group_scores = ops.squeeze(
                ops.sum(
                    ops.top_k(
                        scores_for_choice.reshape(
                            (bsz_seq_len, self.n_group, -1)
                        ),
                        2,
                        axis=-1,
                    )[0],
                    axis=-1,
                ),
                -1,
            )  # [n, n_group]

            # Note: In the original implementation, none of these top-k
            # calls are sorted. We currently only support sorted=True (has no
            # effect on the output of the MoE layer).
            group_idx = ops.top_k(group_scores, k=self.topk_group, axis=-1)[
                1
            ]  # [n, top_k_group]

            group_mask = _fill(
                False, DType.bool, group_scores.shape, group_scores.device
            )
            update = _fill(True, DType.bool, group_idx.shape, group_idx.device)
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
                scores_for_choice,
                ops.constant(
                    float("-inf"), dtype=DType.float32, device=score_mask.device
                ),
            )  # [n, e]
            _, topk_idx = ops.top_k(tmp_scores, k=self.top_k, axis=-1)
            topk_weight = ops.gather_nd(scores, ops.unsqueeze(topk_idx, 2), 1)

        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = ops.sum(topk_weight, axis=-1) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = (
            topk_weight * self.routed_scaling_factor
        )  # must multiply the scaling factor

        return topk_idx, topk_weight

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        if strategy.is_replicate:
            self._sharding_strategy = strategy
            self.gate_score.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.e_score_correction_bias.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
        else:
            raise ValueError(
                "Only replicate sharding strategy is supported for MoEGate."
            )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded DeepseekV3TopKRouter instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = DeepseekV3TopKRouter(
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.scoring_func,
                topk_method=self.topk_method,
                n_group=self.n_group,
                topk_group=self.topk_group,
                norm_topk_prob=self.norm_topk_prob,
                dtype=self.dtype,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                devices=[device],
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
