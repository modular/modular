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

"""GPT OSS Mixture of Experts Layer."""

from __future__ import annotations

from copy import copy

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.clamp import clamp
from max.nn.linear import Linear
from max.nn.moe import GateUpFormat, MoEGate, StackedMoE
from typing_extensions import Self

from ..model_config import GptOssConfig

# GptOss activation parameters
_GPTOSS_ALPHA = 1.702
_GPTOSS_LIMIT = 7.0


def gptoss_glu_activation(gate: TensorValue, up: TensorValue) -> TensorValue:
    """GptOss-style GLU activation with clamping.

    Computes: (up + 1) * gate * sigmoid(gate * alpha)
    with clamping: gate clamped to max=limit, up clamped to [-limit, limit]
    """
    gate = ops.min(gate, _GPTOSS_LIMIT)
    up = clamp(up, min=-_GPTOSS_LIMIT, max=_GPTOSS_LIMIT)
    glu = gate * ops.sigmoid(gate * _GPTOSS_ALPHA)
    return (up + 1.0) * glu


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
    ) -> None:
        """
        Args:
            devices: List of devices to use for the MoEGate.
            hidden_dim: The dimension of the hidden state.
            num_experts: The number of experts.
            num_experts_per_token: The number of experts per token.
            dtype: The data type of the MoEGate.
        """
        # Initialize parent class
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
        )

        # Override gate_score with bias-enabled Linear layer
        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            dtype=dtype,
            device=devices[0],
            has_bias=True,  # Enable bias
        )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """
        Args:
            hidden_state: The hidden state of the model.

        Returns:
            A tuple of the topk indices and scores with softmax applied.
        """
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )

        # Apply softmax to top-k scores (matching GptOss behavior)
        topk_scores = ops.softmax(topk_scores)

        return topk_indices, topk_scores


class GptOssMoE(StackedMoE):
    """GptOss-style MoE implementation with custom activation and biases.

    Inherits from StackedMoE with interleaved gate/up format and custom GLU activation.
    """

    def __init__(
        self,
        config: GptOssConfig,
        is_sharding: bool = False,
    ):
        """
        Args:
            config: The configuration for the GPT OSS Model.
            is_sharding: Set by shard() to skip weight init for sharded instances.
        """
        self.config = config

        super().__init__(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=GptOssMoEGate,
            dtype=config.dtype,
            gate_up_format=GateUpFormat.INTERLEAVED,
            activation_fn=gptoss_glu_activation,
            has_bias=True,
            is_sharding=is_sharding,
        )

    def _create_sharded_instance(
        self, device: DeviceRef, sharded_moe_dim: int, sharded_shared_dim: int
    ) -> Self:
        """Factory method to create a sharded instance using config.

        Overrides base class to use config-based initialization.
        """
        new_config = copy(self.config)
        new_config.devices = [device]
        new_config.intermediate_size = sharded_moe_dim

        return self.__class__(
            config=new_config,
            is_sharding=True,
        )
