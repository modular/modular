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
"""Mixture of Experts Layer for Qwen3VL MoE."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.legacy.float8_config import Float8Config
from max.nn.legacy.kernels import (
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    moe_create_indices,
    quantize_dynamic_scaled_float8,
)
from max.nn.legacy.moe import MoE, MoEGate
from max.support.math import ceildiv


def _compute_shard_range(
    shard_dim: int, shard_idx: int, num_devices: int
) -> tuple[int, int]:
    """Compute the start and end indices for a shard.

    Args:
        shard_dim: The dimension to shard.
        shard_idx: The index of the shard.
        num_devices: The total number of devices.

    Returns:
        A tuple of (start, end) indices for the shard.
    """
    base_size, remainder = divmod(shard_dim, num_devices)

    # Give first 'remainder' devices an extra column/row.
    if shard_idx < remainder:
        start = shard_idx * (base_size + 1)
        end = start + base_size + 1
    else:
        start = (
            remainder * (base_size + 1) + (shard_idx - remainder) * base_size
        )
        end = start + base_size

    return start, end


def gate_up_scale_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    moe_dim: int,
    block_size: int,
    axis: int = 2,
) -> TensorValue:
    """Shards a combined gate/up projection scale tensor.

    This strategy properly maps weight shard indices to scale indices,
    accounting for the block size used in FP8 quantization. Unlike the
    generic gate_up sharding, this ensures the scale slices correspond
    exactly to the sharded weight elements.

    Args:
        weight: The scale :obj:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        moe_dim: The original moe_dim (half of the gate_up weight dimension).
        block_size: The block size used for quantization scaling.
        axis: The axis along which the gate and up scales are concatenated.

    Returns:
        A :obj:`TensorValue` representing the sharded scale for the i-th device.
    """
    # Compute the weight shard range within each half (gate and up)
    weight_start, weight_end = _compute_shard_range(moe_dim, i, num_devices)

    # Map weight indices to scale indices
    # For the gate portion: weight indices [weight_start, weight_end)
    scale_gate_start = weight_start // block_size
    scale_gate_end = ceildiv(weight_end, block_size)

    # For the up portion: weight indices [moe_dim + weight_start, moe_dim + weight_end)
    scale_up_start = (moe_dim + weight_start) // block_size
    scale_up_end = ceildiv(moe_dim + weight_end, block_size)

    rank = len(weight.shape)
    if axis < 0:
        axis += rank

    # Create slices for gate scale
    gate_slices = [slice(None)] * rank
    gate_slices[axis] = slice(scale_gate_start, scale_gate_end)

    # Create slices for up scale
    up_slices = [slice(None)] * rank
    up_slices[axis] = slice(scale_up_start, scale_up_end)

    sharded_gate_scale = weight[tuple(gate_slices)]
    sharded_up_scale = weight[tuple(up_slices)]

    return ops.concat((sharded_gate_scale, sharded_up_scale), axis=axis)


def down_proj_scale_sharding_strategy(
    weight: Weight,
    i: int,
    num_devices: int,
    moe_dim: int,
    block_size: int,
    axis: int = 1,
) -> TensorValue:
    """Shards a down projection scale tensor along axis.

    This strategy properly maps weight shard indices to scale indices,
    accounting for the block size used in FP8 quantization. Unlike the
    generic axiswise sharding, this ensures the scale slices correspond
    exactly to the sharded weight elements.

    Args:
        weight: The scale :obj:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.
        moe_dim: The original moe_dim (the weight dimension being sharded).
        block_size: The block size used for quantization scaling.
        axis: The axis along which to shard.

    Returns:
        A :obj:`TensorValue` representing the sharded scale for the i-th device.
    """
    # Compute the weight shard range
    weight_start, weight_end = _compute_shard_range(moe_dim, i, num_devices)

    # Map weight indices to scale indices
    scale_start = weight_start // block_size
    scale_end = ceildiv(weight_end, block_size)

    rank = len(weight.shape)
    if axis < 0:
        axis += rank

    # Create slices for the scale
    slices = [slice(None)] * rank
    slices[axis] = slice(scale_start, scale_end)

    return weight[tuple(slices)]


class Qwen3VLMoEGate(MoEGate):
    """Qwen3VL MoE Gate with simple top-k routing and softmax normalization."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        is_sharding: bool = False,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoEGate.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            dtype: The data type of the MoEGate.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=is_sharding,
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
        # Compute router logits
        router_logits = self.gate_score(hidden_states).cast(DType.float32)
        # Apply softmax to get routing weights
        routing_weights = ops.softmax(router_logits, axis=-1)
        # Select top-k experts
        topk_weights, topk_indices = ops.top_k(
            routing_weights, k=self.num_experts_per_token, axis=-1
        )

        # Normalize top-k weights to sum to 1
        denominator = ops.sum(topk_weights, axis=-1) + 1e-20

        topk_weights = topk_weights / denominator
        # Cast back to original dtype
        topk_weights = topk_weights.cast(hidden_states.dtype)
        return topk_indices, topk_weights

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
        else:
            raise ValueError(
                "Only replicate sharding strategy is supported for MoEGate."
            )

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MoEGate]:
        """Create sharded views of this MoEGate module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Qwen3VLMoEGate instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoEGate module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_score_shards = self.gate_score.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = Qwen3VLMoEGate(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                dtype=self.dtype,
                is_sharding=True,
            )

            # Replace the weights with sharded versions.
            sharded.gate_score = gate_score_shards[shard_idx]
            shards.append(sharded)
        return shards


class Qwen3VLMoE(MoE):
    """Qwen3VL MoE implementation with standard gated activation."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate] = Qwen3VLMoEGate,
        dtype: DType = DType.bfloat16,
        mlp_only_layers: list[int] | None = None,
        float8_config: Float8Config | None = None,
        is_sharding: bool = False,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoE.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            dtype: The data type of the MoE.
            mlp_only_layers: List of layer indices that use MLP instead of MoE (unused here, kept for compatibility).
            float8_config: Configuration for FP8 quantization.
            is_sharding: Whether the module is being sharded.
        """
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=gate_cls,
            dtype=dtype,
            has_shared_experts=False,
            shared_experts_dim=0,
            ep_size=1,
            apply_router_weight_first=False,
            ep_batch_manager=None,
            float8_config=float8_config,
            is_sharding=is_sharding,
        )
        self.mlp_only_layers = mlp_only_layers

    def _init_experts(self) -> None:
        """Initialize experts using stacked weight tensors instead of individual MLPs.

        This matches how weights are stored in the checkpoint:
        - layers.{i}.mlp.experts.gate_up_proj with shape [num_experts, hidden_dim, 2*moe_dim]
        - layers.{i}.mlp.experts.down_proj with shape [num_experts, moe_dim, hidden_dim]
        """
        # Create stacked weight tensors for all experts
        # Gate and up projections are combined into one tensor
        self._experts_gate_up_proj_weight = Weight(
            "experts.gate_up_proj",
            shape=[self.num_experts, self.hidden_dim, 2 * self.moe_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Down projection weights
        self._experts_down_proj_weight = Weight(
            "experts.down_proj",
            shape=[self.num_experts, self.moe_dim, self.hidden_dim],
            dtype=self.dtype,
            device=self.devices[0],
        )

        # Gate, Up, and Down projection scales for FP8 quantization.
        if self.float8_config:
            block_size = self.float8_config.weight_scale.block_size
            assert block_size is not None, "FP8 MoE requires block scaling"

            gate_up_scale_shape = [
                self.num_experts,
                ceildiv(self.hidden_dim, block_size[0]),
                ceildiv(2 * self.moe_dim, block_size[1]),
            ]

            down_scale_shape = [
                self.num_experts,
                ceildiv(self.moe_dim, block_size[0]),
                ceildiv(self.hidden_dim, block_size[1]),
            ]

            self._experts_gate_up_proj_weight_scale = Weight(
                "experts.gate_up_proj_scale",
                shape=gate_up_scale_shape,
                dtype=self.float8_config.weight_scale.dtype,
                device=self.devices[0],
            )
            self._experts_down_proj_weight_scale = Weight(
                "experts.down_proj_scale",
                shape=down_scale_shape,
                dtype=self.float8_config.weight_scale.dtype,
                device=self.devices[0],
            )

    @property
    def gate_up_proj(self) -> TensorValue:
        """Return combined gate_up projection weights for grouped_matmul_ragged.

        grouped_matmul_ragged expects shape [num_experts, out_features, in_features].
        """
        # Return the combined gate_up projection weights, transposed for grouped_matmul_ragged
        # grouped_matmul_ragged expects shape [num_experts, out_features, in_features]
        return self._experts_gate_up_proj_weight.transpose(1, 2)

    @property
    def down_proj(self) -> TensorValue:
        """Return down projection weights for grouped_matmul_ragged.

        grouped_matmul_ragged expects shape [num_experts, out_features, in_features].
        """
        # Transpose for grouped_matmul_ragged: [num_experts, hidden_dim, moe_dim]
        return self._experts_down_proj_weight.transpose(1, 2)

    @property
    def gate_up_proj_scale(self) -> TensorValue:
        """Return combined gate_up projection scales for grouped_dynamic_scaled_fp8_matmul."""
        return self._experts_gate_up_proj_weight_scale.transpose(1, 2)

    @property
    def down_proj_scale(self) -> TensorValue:
        """Return down projection scales for grouped_dynamic_scaled_fp8_matmul."""
        return self._experts_down_proj_weight_scale.transpose(1, 2)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for the module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the module."""
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy
            self.gate.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            # Set sharding strategy for the stacked expert weights
            self._experts_gate_up_proj_weight.sharding_strategy = (
                ShardingStrategy.gate_up(strategy.num_devices)
            )
            self._experts_down_proj_weight.sharding_strategy = (
                ShardingStrategy.axiswise(
                    axis=1, num_devices=strategy.num_devices
                )
            )

            if self.float8_config:
                # Use custom scale sharding that properly maps weight indices
                # to scale indices based on block size
                block_size = self.float8_config.weight_scale.block_size
                assert block_size is not None, "FP8 MoE requires block scaling"

                # Create custom sharding strategy for gate_up scale
                gate_up_scale_shard_fn = partial(
                    gate_up_scale_sharding_strategy,
                    moe_dim=self.moe_dim,
                    block_size=block_size[1],
                    axis=2,
                )
                self._experts_gate_up_proj_weight_scale.sharding_strategy = (
                    ShardingStrategy(
                        num_devices=strategy.num_devices,
                        shard=gate_up_scale_shard_fn,
                    )
                )

                # Create custom sharding strategy for down_proj scale
                # Down proj weight is sharded on axis=1 (moe_dim), so scale
                # needs to map weight indices to scale indices using block_size[0]
                down_proj_scale_shard_fn = partial(
                    down_proj_scale_sharding_strategy,
                    moe_dim=self.moe_dim,
                    block_size=block_size[0],
                    axis=1,
                )
                self._experts_down_proj_weight_scale.sharding_strategy = (
                    ShardingStrategy(
                        num_devices=strategy.num_devices,
                        shard=down_proj_scale_shard_fn,
                    )
                )
        else:
            raise ValueError(
                "Only tensor parallel sharding strategy is supported for Qwen3VLMoE"
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """
        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)

        router_idx = ops.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)
        router_idx_int32 = ops.cast(router_idx, DType.int32)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_int32, self.num_experts)

        # Extract token indices from token_expert_order
        # token_expert_order contains indices into the flattened router_idx array
        # which has shape [seq_len * num_experts_per_token]
        # To get the token index, we divide by num_experts_per_token
        token_indices = ops.cast(
            token_expert_order // self.num_experts_per_token, DType.int32
        )

        permutated_states = ops.gather(x, token_indices, axis=0)

        if self.float8_config:
            # FP8 Path
            assert self.float8_config.input_scale.block_size is not None
            input_block_size = self.float8_config.input_scale.block_size[1]

            # Ensure input is BF16 for dynamic quantization
            if permutated_states.dtype != DType.bfloat16:
                raise ValueError("Input must be BF16 for dynamic quantization")

            # 1. Quantize Input
            permutated_states_fp8, permutated_states_scales = (
                quantize_dynamic_scaled_float8(
                    permutated_states,
                    self.float8_config.input_scale,
                    self.float8_config.weight_scale,
                    group_size_or_per_token=input_block_size,
                    out_type=self.dtype,
                    scales_type=self.float8_config.weight_scale.dtype,
                )
            )

            # 2. Gate + Up Projection
            gate_up_projs = grouped_dynamic_scaled_fp8_matmul(
                permutated_states_fp8,
                self.gate_up_proj,
                permutated_states_scales,
                self.gate_up_proj_scale,
                expert_start_indices,
                expert_ids,
                expert_usage_stats.to(DeviceRef.CPU()),
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
            )

            # Activation computed in BF16
            if gate_up_projs.dtype != DType.bfloat16:
                raise ValueError(
                    "Gate + Up projection output must be BF16 for dynamic quantization"
                )

            up = gate_up_projs[:, self.moe_dim :]
            gate = gate_up_projs[:, : self.moe_dim]
            gate_up_projs = up * ops.silu(gate)

            # 3. Quantize Intermediate
            gate_up_projs_fp8, gate_up_projs_scales = (
                quantize_dynamic_scaled_float8(
                    gate_up_projs,
                    self.float8_config.input_scale,
                    self.float8_config.weight_scale,
                    group_size_or_per_token=input_block_size,
                    out_type=self.dtype,
                    scales_type=self.float8_config.weight_scale.dtype,
                )
            )

            # 4. Down Projection
            down_projs = grouped_dynamic_scaled_fp8_matmul(
                gate_up_projs_fp8,
                self.down_proj,
                gate_up_projs_scales,
                self.down_proj_scale,
                expert_start_indices,
                expert_ids,
                expert_usage_stats.to(DeviceRef.CPU()),
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
            )
        else:
            # BF16 path
            # Gate + Up projection
            gate_up_projs = grouped_matmul_ragged(
                permutated_states,
                self.gate_up_proj,
                expert_start_indices,
                expert_ids,
                expert_usage_stats.to(DeviceRef.CPU()),
            )

            # Standard gated activation: up * act_fn(gate)
            # Split gate and up projections (concatenated format)
            up = gate_up_projs[:, self.moe_dim :]
            gate = gate_up_projs[:, : self.moe_dim]

            gated_output = up * ops.silu(gate)

            # Down projection
            down_projs = grouped_matmul_ragged(
                gated_output,
                self.down_proj,
                expert_start_indices,
                expert_ids,
                expert_usage_stats.to(DeviceRef.CPU()),
            )

        # Restore token order and reshape
        down_projs = ops.gather(
            down_projs, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, self.hidden_dim])

        # Weighted combination: (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down_projs
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(x.dtype)

        return routed_expert_out

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3VLMoE]:
        """Create sharded views of this MoE module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MoE instances, one for each device."""
        if not self._sharding_strategy:
            raise ValueError(
                "MoE module cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        gate_shards = self.gate.shard(devices)

        # Shard the stacked expert weight tensors
        experts_gate_up_proj_shards = self._experts_gate_up_proj_weight.shard(
            devices
        )
        experts_down_proj_shards = self._experts_down_proj_weight.shard(devices)

        if self.float8_config:
            experts_gate_up_proj_scale_shards = (
                self._experts_gate_up_proj_weight_scale.shard(devices)
            )
            experts_down_proj_scale_shards = (
                self._experts_down_proj_weight_scale.shard(devices)
            )

        shards = []
        num_devices = self._sharding_strategy.num_devices
        sharded_moe_dim = self.moe_dim // num_devices
        for shard_idx, device in enumerate(devices):
            sharded = self.__class__(
                devices=[device],
                hidden_dim=self.hidden_dim,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                moe_dim=sharded_moe_dim,
                gate_cls=self.gate_cls,
                dtype=self.dtype,
                mlp_only_layers=self.mlp_only_layers,
                float8_config=self.float8_config,
                is_sharding=True,
            )
            # Replace layers and weights with sharded versions.
            sharded.gate = gate_shards[shard_idx]

            # Replace the stacked expert weights with sharded versions
            sharded._experts_gate_up_proj_weight = experts_gate_up_proj_shards[
                shard_idx
            ]
            sharded._experts_down_proj_weight = experts_down_proj_shards[
                shard_idx
            ]

            if self.float8_config:
                sharded._experts_gate_up_proj_weight_scale = (
                    experts_gate_up_proj_scale_shards[shard_idx]
                )
                sharded._experts_down_proj_weight_scale = (
                    experts_down_proj_scale_shards[shard_idx]
                )

            shards.append(sharded)

        return shards
