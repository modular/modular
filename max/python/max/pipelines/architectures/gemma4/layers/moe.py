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
"""Gemma4 Mixture of Experts."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.moe.moe import MoEGate
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm


class Gemma4MoEGate(MoEGate):
    """Gemma4 MoE gate with RMSNorm, learned scale, and per-expert scaling."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType = DType.bfloat16,
        is_sharding: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
            is_sharding=is_sharding,
        )
        self.scalar_root_size = hidden_dim**-0.5
        self.eps = eps

        if not is_sharding:
            self.norm = Gemma4RMSNorm(
                dim=hidden_dim, dtype=dtype, eps=eps, with_weight=False
            )
            self.scale = Weight(
                name="scale", dtype=dtype, shape=[hidden_dim], device=devices[0]
            )
            self.per_expert_scale = Weight(
                name="per_expert_scale",
                dtype=dtype,
                shape=[num_experts],
                device=devices[0],
            )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state * self.scale * self.scalar_root_size

        expert_scores = self.gate_score(hidden_state)
        router_probs = ops.softmax(expert_scores)

        top_k_weights, top_k_index = ops.top_k(
            router_probs, k=self.num_experts_per_token, axis=-1
        )

        top_k_weights = top_k_weights / ops.sum(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * ops.gather(
            self.per_expert_scale, top_k_index, axis=0
        )

        return top_k_index, top_k_weights
