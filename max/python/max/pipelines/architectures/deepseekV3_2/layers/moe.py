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
"""Mixture of Experts (MoE) module for DeepseekV3.2."""

from __future__ import annotations

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy import Float8Config
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.legacy.comm.ep.ep_kernels import fused_silu
from max.nn.legacy.kernels import grouped_matmul_ragged, moe_create_indices
from max.nn.legacy.layer import LayerList
from max.nn.legacy.moe import MoE as BaseMoE
from max.nn.legacy.moe import MoEGate

from .mlp import MLP


class MoE(BaseMoE):
    """Mixture of Experts module for DeepSeek V3.2.

    Key differences from base MoE (V3):
    - Intermediate activations cast to float32 for numerical stability
    - Expert outputs accumulated in float32
    - Final output cast back to original dtype
    """

    shared_experts: MLP  # type: ignore[assignment]

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        moe_dim: int,
        gate_cls: Callable[..., MoEGate],
        has_shared_experts: bool = False,
        shared_experts_dim: int = 0,
        ep_size: int = 1,
        dtype: DType = DType.bfloat16,
        apply_router_weight_first: bool = False,
        ep_batch_manager: EPBatchManager | None = None,
        float8_config: Float8Config | None = None,
        is_sharding: bool = False,
    ):
        """
        Args:
            devices: List of devices to use for the MoE.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            moe_dim: The intermediate dimension of each expert.
            gate_cls: The model specific gate implementation.
            has_shared_experts: Whether to use shared experts.
            shared_experts_dim: The dimension of the shared experts.
            ep_size: The expert parallelism size.
            dtype: The data type of the MoE.
            apply_router_weight_first: Whether to apply the router weight first.
            ep_batch_manager: The expert parallel batch manager.
            float8_config: Optional Float8Config for float8 quantization.
            is_sharding: Disable child layer creation during sharding.
        """
        # Initialize base class (we override _init_experts to use our MLP)
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_dim=moe_dim,
            gate_cls=gate_cls,
            has_shared_experts=False,
            shared_experts_dim=shared_experts_dim,
            ep_size=ep_size,
            dtype=dtype,
            apply_router_weight_first=apply_router_weight_first,
            ep_batch_manager=ep_batch_manager,
            float8_config=float8_config,
            is_sharding=is_sharding,
        )

        self.has_shared_experts = has_shared_experts

        # Override shared experts to use V3.2 MLP with float32 intermediate ops.
        if has_shared_experts:
            self.shared_experts = MLP(
                dtype=dtype,
                hidden_dim=hidden_dim,
                feed_forward_length=shared_experts_dim,
                device=devices[0],
                float8_config=float8_config,
            )

    def _init_experts(self) -> None:
        """Override to use MLP with float32 intermediate operations."""
        self.experts = LayerList(
            [
                MLP(
                    dtype=self.dtype,
                    hidden_dim=self.hidden_dim,
                    feed_forward_length=self.moe_dim,
                    device=self.devices[0],
                    float8_config=self.float8_config,
                )
                for _ in range(self.num_experts)
            ]
        )

    def _ep_call(
        self,
        x: TensorValue,
        router_idx: TensorValue,
        router_weight: TensorValue,
    ) -> TensorValue:
        """Expert parallel forward pass with float32 intermediate operations."""
        device_id = self.devices[0].id
        expert_inputs = self.ep_batch_manager.ep_dispatch(
            x, router_idx, device_id
        )

        # Cast to float32 for intermediate operations
        gate_up_projs = grouped_matmul_ragged(
            expert_inputs[0],
            self.gate_up_proj,
            *expert_inputs[1:],
        ).cast(DType.float32)

        silu_out = fused_silu(
            gate_up_projs.cast(x.dtype), expert_inputs[1]
        ).cast(DType.float32)

        down_projs = grouped_matmul_ragged(
            silu_out.cast(x.dtype),
            self.down_proj,
            *expert_inputs[1:],
        ).cast(DType.float32)

        routed_expert_out = self.ep_batch_manager.ep_combine(
            down_projs.cast(x.dtype), router_weight, device_id
        ).cast(DType.float32)

        if (
            self.has_shared_experts
            and not self.ep_batch_manager.config.fused_shared_expert
        ):
            routed_expert_out = routed_expert_out + self.shared_experts(x).cast(
                DType.float32
            )

        return routed_expert_out.cast(x.dtype)

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass with float32 intermediate operations.

        Args:
            x: (seq_len, hidden_dim)

        Returns:
            (seq_len, hidden_dim)
        """
        seq_len = x.shape[0]
        original_dtype = x.dtype

        # Get the topk experts per token and their weights
        router_idx, router_weight = self.gate(x)
        if self._ep_batch_manager:
            return self._ep_call(
                x, ops.cast(router_idx, DType.int32), router_weight
            )

        router_idx = ops.reshape(
            router_idx, [-1]
        )  # (seq_len * n_expert_per_token,)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            ops.cast(
                token_expert_order // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * ops.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x.dtype)

        # Cast to float32 for intermediate operations (V3.2 change)
        gate_up_projs = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        ).cast(DType.float32)

        gate_up_projs = (
            ops.silu(gate_up_projs[:, : self.moe_dim])
            * gate_up_projs[:, self.moe_dim :]
        )

        down_projs = grouped_matmul_ragged(
            gate_up_projs.cast(original_dtype),
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats.to(DeviceRef.CPU()),
        ).cast(DType.float32)

        down_projs = ops.gather(
            down_projs, restore_token_order, axis=0
        ).reshape([seq_len, self.num_experts_per_token, self.hidden_dim])

        if not self.apply_router_weight_first:
            # (seq_len, 1, n_expert) @ (seq_len, n_expert, hidden_dim) -> (seq_len, 1, hidden_dim)
            routed_expert_out = (
                ops.unsqueeze(router_weight, axis=1) @ down_projs
            )
            routed_expert_out = ops.squeeze(routed_expert_out, axis=1)
        else:
            routed_expert_out = down_projs.transpose(1, 2)
            routed_expert_out = ops.squeeze(
                ops.sum(routed_expert_out, axis=2), axis=2
            )

        # Add shared experts (accumulate in float32)
        if self.has_shared_experts:
            routed_expert_out = routed_expert_out + self.shared_experts(x).cast(
                DType.float32
            )

        # Cast back to original dtype
        return routed_expert_out.cast(original_dtype)


class MoEFp8(MoE):
    """FP8 variant of MoE for DeepSeek V3.2.

    Uses the same float32 intermediate operations as MoE but with FP8 quantization.
    """

    pass
