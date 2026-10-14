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
"""MiMo-V2 routed experts: stacked MXFP4 weights, run W4A8."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.moe import SigmoidTopKRouter, StackedMoE
from max.nn.quant_config import QuantConfig


def mimo_v2_moe(
    *,
    hidden_dim: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_dim: int,
    norm_topk_prob: bool,
    quant_config: QuantConfig,
    devices: list[DeviceRef],
) -> StackedMoE:
    """Returns MiMo-V2's sigmoid-routed MoE, with no shared expert.

    The router runs in float32 from float32 weights, as the reference does;
    the correction bias steers selection only, and the combine weights are
    the unbiased sigmoid scores renormalized over the top k, accumulated with
    the expert outputs in float32. The experts run W4A8 from stacked MXFP4
    weights, gate rows then up rows.
    """
    return StackedMoE(
        devices=devices,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_tok,
        moe_dim=moe_dim,
        gate_cls=functools.partial(
            SigmoidTopKRouter, norm_topk_prob=norm_topk_prob
        ),
        quant_config=quant_config,
        router_dtype=DType.float32,
        mxfp8_activations=True,
    )
