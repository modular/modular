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
"""Mixture-of-experts feed-forward block for the DSpark drafter's stages.

Each stage routes draft tokens to a few quantized experts and adds one
always-active shared expert, with its own expert counts separate from the
target model's.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.layer import Module

from ..model_config import DeepseekV41Config
from .hyper_connection import Mxfp8Rows
from .moe import (
    DeepseekV41Gate,
    DeepseekV41Mxfp4RoutedExperts,
    DeepseekV41SharedExpert,
)


class DeepseekV41DSparkMoE(Module):
    """One DSpark stage's FFN: gate, MXFP4 routed experts, shared expert.

    Drafts are text, so the gate always uses its plain ``bias`` (never the
    image-token ``bias_vl``).
    """

    def __init__(self, config: DeepseekV41Config, device: DeviceRef) -> None:
        super().__init__()
        if config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek-V4.1 assumes exactly one shared expert, got "
                f"{config.n_shared_experts}"
            )
        self.gate = DeepseekV41Gate(
            n_routed_experts=config.dspark_n_routed_experts,
            num_experts_per_tok=config.dspark_num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            hidden_size=config.hidden_size,
            dtype=DType.bfloat16,
            device=device,
        )
        self.experts = DeepseekV41Mxfp4RoutedExperts(
            n_experts=config.dspark_n_routed_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.dspark_num_experts_per_tok,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )
        self.shared_experts = DeepseekV41SharedExpert(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )

    def shard_experts(self, num_devices: int) -> None:
        """Declares the expert-parallel split of the routed experts.

        Each device owns a contiguous run of whole experts; the gate and the
        shared expert are replicated.

        Raises:
            ValueError: If the experts do not divide across ``num_devices``.
        """
        n_experts = self.experts.n_global_experts
        if n_experts % num_devices:
            raise ValueError(
                f"{n_experts} DSpark routed experts do not split across "
                f"{num_devices} devices"
            )
        self.experts.shard_experts(num_devices)

    def keep_local_experts(self, num_devices: int, rank: int) -> None:
        """Makes this replica compute only its run of :meth:`shard_experts`.

        :meth:`routed` then returns this device's float32 partial sum.
        """
        self.experts.keep_local_experts(num_devices, rank)

    def routed(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """Returns this replica's float32 ``[b, s, hidden]`` routed output."""
        # Any non-image id: drafts are text.
        text_ids = ops.broadcast_to(
            ops.constant(0, token_ids.dtype, token_ids.device),
            token_ids.shape,
        )
        weights, indices = self.gate(x, text_ids)
        hidden = int(x.shape[-1])
        batch, seq = x.shape[0], x.shape[1]
        routed = self.experts(
            ops.reshape(x, [batch * seq, hidden]),
            ops.reshape(weights, [batch * seq, self.experts.topk]),
            ops.reshape(indices, [batch * seq, self.experts.topk]),
        )
        return ops.reshape(
            ops.cast(routed, DType.float32), [batch, seq, hidden]
        )

    def __call__(
        self,
        x: TensorValue,
        token_ids: TensorValue,
        x_mxfp8: Mxfp8Rows | None = None,
    ) -> TensorValue:
        """Maps ``[1, s, hidden]`` states and ``[1, s]`` ids to the output.

        The shared expert reads ``x_mxfp8`` (``x`` already in MXFP8) if given.
        """
        y = ops.cast(
            self.shared_experts(x, x_mxfp8), DType.float32
        ) + self.routed(x, token_ids)
        return ops.cast(y, x.dtype)
