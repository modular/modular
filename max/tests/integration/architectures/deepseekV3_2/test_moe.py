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
"""Tests for DeepseekV3.2 MoE layer."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.linear import Linear
from max.pipelines.architectures.deepseekV3_2.layers import (
    DeepseekV3_2MLP,
    DeepseekV3_2MoE,
    DeepseekV3_2TopKRouter,
)


def test_moe_with_shared_experts() -> None:
    """Tests MoE with shared experts."""
    hidden_dim = 64
    num_experts = 8
    num_experts_per_token = 2
    moe_dim = 128
    shared_experts_dim = 256

    moe = DeepseekV3_2MoE(
        devices=[DeviceRef.CPU()],
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        moe_dim=moe_dim,
        gate_cls=functools.partial(
            DeepseekV3_2TopKRouter,
            routed_scaling_factor=1.0,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            norm_topk_prob=False,
            gate_dtype=DType.bfloat16,
            correction_bias_dtype=DType.float32,
            linear_cls=Linear,
        ),
        mlp_cls=DeepseekV3_2MLP,
        has_shared_experts=True,
        shared_experts_dim=shared_experts_dim,
        dtype=DType.bfloat16,
        apply_router_weight_first=False,
    )

    # Initialize weight names to avoid collisions.
    moe.state_dict()

    def forward(x: TensorValue) -> TensorValue:
        return moe(x)

    g = Graph(
        "test_moe_with_shared_experts",
        forward=forward,
        input_types=[
            TensorType(DType.bfloat16, (10, hidden_dim), DeviceRef.CPU())
        ],
    )

    # Verify the graph was constructed successfully.
    assert g is not None
    # Verify shared experts were created
    assert moe.has_shared_experts
    assert moe.shared_experts is not None


def test_moe_gate() -> None:
    """Tests MoE gate routing."""
    hidden_dim = 64
    num_experts = 8
    num_experts_per_token = 2

    gate = DeepseekV3_2TopKRouter(
        num_experts_per_token=num_experts_per_token,
        num_experts=num_experts,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        hidden_dim=hidden_dim,
        dtype=DType.bfloat16,
        gate_dtype=DType.bfloat16,
        correction_bias_dtype=DType.float32,
        devices=[DeviceRef.CPU()],
        linear_cls=Linear,  # Will be replaced by FloatCastingLinear
    )

    # Initialize weight names to avoid collisions.
    gate.state_dict()

    def forward(x: TensorValue) -> tuple[TensorValue, TensorValue]:
        return gate(x)

    g = Graph(
        "test_moe_gate",
        forward=forward,
        input_types=[
            TensorType(DType.bfloat16, (10, hidden_dim), DeviceRef.CPU())
        ],
    )

    # Verify the graph was constructed successfully.
    assert g is not None
