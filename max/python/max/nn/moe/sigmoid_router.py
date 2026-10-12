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
"""Sigmoid top-k MoE router with an expert score correction bias."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from typing_extensions import Self

from ..kernels import moe_router_group_limited
from ..linear import Linear
from .moe import MoEGate


class SigmoidTopKRouter(MoEGate):
    """MoE gate with sigmoid scores and a correction bias (``noaux_tc``).

    Routes as DeepSeek-V3's auxiliary-loss-free ``noaux_tc`` gate does with a
    single expert group:

    1. Computes the gate logits with a linear projection.
    2. Applies a sigmoid, in ``correction_bias_dtype``.
    3. Selects the top-k experts by the sigmoid score plus the learned
       ``e_score_correction_bias``.
    4. Weights each selected expert by its unbiased sigmoid score, optionally
       renormalized to sum to 1, times ``routed_scaling_factor``.

    For a float32 router, pass ``dtype=DType.float32`` and a float32 input:
    the gate projection and the scores then stay in float32.

    Args:
        devices: The devices to place the gate on.
        hidden_dim: The dimension of the hidden state.
        num_experts: The number of routed experts.
        num_experts_per_token: The number of experts each token selects.
        dtype: The gate projection's dtype, unless ``gate_dtype`` is set.
        norm_topk_prob: Whether to renormalize the selected weights to sum
            to 1. Defaults to ``True``.
        gate_dtype: The gate projection's dtype. Defaults to ``dtype``.
        correction_bias_dtype: The dtype of the correction bias and of the
            sigmoid scores. Defaults to ``DType.float32``.
        routed_scaling_factor: The factor applied to the routing weights.
            Defaults to ``1.0``.
        linear_cls: The linear class for the gate projection.
        is_sharding: Whether :meth:`shard` is constructing this gate, which
            then assigns the weights itself.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        norm_topk_prob: bool = True,
        gate_dtype: DType | None = None,
        correction_bias_dtype: DType = DType.float32,
        routed_scaling_factor: float = 1.0,
        linear_cls: Callable[..., Linear] = Linear,
        is_sharding: bool = False,
    ) -> None:
        gate_dtype = gate_dtype or dtype
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=gate_dtype,
            linear_cls=linear_cls,
            is_sharding=is_sharding,
        )
        self.norm_topk_prob = norm_topk_prob
        self.gate_dtype = gate_dtype
        self.correction_bias_dtype = correction_bias_dtype
        self.routed_scaling_factor = routed_scaling_factor
        self.linear_cls = linear_cls
        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[self.num_experts],
            device=self.devices[0],
            dtype=correction_bias_dtype,
        )

    def _gate_logits(self, hidden_states: TensorValue) -> TensorValue:
        """Returns the ``[seq_len, num_experts]`` router logits.

        Subclasses override this to compute the projection another way, for
        example in a wider dtype than the stored weight.
        """
        return self.gate_score(hidden_states)

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Routes each token to its top-k experts.

        Args:
            hidden_states: The ``[seq_len, hidden_dim]`` router input.

        Returns:
            A tuple ``(topk_idx, topk_weight)``, each
            ``[seq_len, num_experts_per_token]``: the selected experts and
            their routing weights.
        """
        logits = self._gate_logits(hidden_states)
        scores = ops.sigmoid(logits.cast(self.correction_bias_dtype))
        return moe_router_group_limited(
            scores,
            self.e_score_correction_bias,
            self.num_experts,
            self.num_experts_per_token,
            n_groups=1,
            topk_group=1,
            norm_weights=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _set_sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Replicates the correction bias alongside the base gate weights."""
        super()._set_sharding_strategy(strategy)
        self.e_score_correction_bias.sharding_strategy = (
            ShardingStrategy.replicate(strategy.num_devices)
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Self]:
        """Creates one replica of this gate per device.

        Args:
            devices: The devices to place the replicas on.

        Returns:
            One gate per device, of this gate's class.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding "
                "strategy was provided."
            )
        devices = list(devices)
        gate_score_shards = self.gate_score.shard(devices)
        correction_bias_shards = self.e_score_correction_bias.shard(devices)
        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = self.__class__(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                norm_topk_prob=self.norm_topk_prob,
                gate_dtype=self.gate_dtype,
                correction_bias_dtype=self.correction_bias_dtype,
                routed_scaling_factor=self.routed_scaling_factor,
                linear_cls=self.linear_cls,
                is_sharding=True,
            )
            sharded.gate_score = gate_score_shards[shard_idx]
            sharded.e_score_correction_bias = correction_bias_shards[shard_idx]
            shards.append(sharded)
        return shards
