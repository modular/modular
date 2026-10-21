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
"""MiMo-V2 routed experts with MXFP8 activations on NVIDIA GPUs."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.moe import (
    MoEQuantized,
    NvMxf4f8Strategy,
    QuantStrategy,
    SigmoidTopKRouter,
)
from max.nn.quant_config import QuantConfig


class MiMoV2Router(SigmoidTopKRouter):
    """Sigmoid top-k router whose logits are computed in float32.

    The reference scores with ``F.linear(hidden.float(), weight.float())``
    on the BF16 gate weight; the weight's cast runs once, at model init.
    """

    def _gate_logits(self, hidden_states: TensorValue) -> TensorValue:
        hidden = ops.cast(hidden_states, DType.float32)
        weight = ops.cast(self.gate_score.weight, DType.float32).to(
            hidden.device
        )
        return hidden @ ops.transpose(weight, -1, -2)


class MiMoV2MoE(MoEQuantized):
    """Keeps routed-expert activations at MXFP8 precision on NVIDIA GPUs."""

    def _strategy(self) -> QuantStrategy:
        strategy = super()._strategy()
        if isinstance(strategy, NvMxf4f8Strategy) and strategy.config.is_mxfp4:
            # MXFP4 inputs give about 5x the expert error of MXFP8 against a
            # float32 reference (test_expert_precision_gpu.py).
            return NvMxf4f8Strategy(strategy.config, DType.float8_e4m3fn)
        return strategy


def mimo_v2_moe(
    *,
    hidden_dim: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_dim: int,
    norm_topk_prob: bool,
    quant_config: QuantConfig,
    devices: list[DeviceRef],
) -> MoEQuantized:
    """Returns MiMo-V2's sigmoid-routed MoE, with no shared expert.

    The correction bias steers selection only; the combine weights are the
    unbiased sigmoid scores renormalized over the top k, accumulated with the
    expert outputs in float32. On NVIDIA, the experts run W4A8: each
    projection's input is quantized to MXFP8 per expert at runtime.
    """
    return MiMoV2MoE(
        devices=devices,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_tok,
        moe_dim=moe_dim,
        gate_cls=functools.partial(MiMoV2Router, norm_topk_prob=norm_topk_prob),
        dtype=DType.uint8,
        quant_config=quant_config,
    )
