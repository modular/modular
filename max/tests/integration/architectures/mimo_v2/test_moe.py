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
"""Tests that MiMo-V2's MoE declares the weights its adapter produces.

The layer's behavior (the W4A8 experts, their tensor-parallel scale split,
and the sigmoid router) is covered by the framework tests in
``max/tests/tests/nn/test_stacked_moe_w4a8.py``.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.mimo_v2.layers.moe import mimo_v2_moe
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from transformers.configuration_utils import PretrainedConfig

EXPERTS, HIDDEN, WIDTH = 2, 256, 512


def test_serving_declares_only_the_expert_stacks() -> None:
    config = PretrainedConfig(
        num_hidden_layers=2,
        moe_layer_freq=[0, 1],
        n_routed_experts=EXPERTS,
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.1.mlp.experts.{e}.{p}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for e in range(EXPERTS)
                for p in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )
    moe = mimo_v2_moe(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_dim=WIDTH,
        norm_topk_prob=True,
        quant_config=parse_quant_scheme(config).experts,
        devices=[DeviceRef.CPU()],
    )
    weights = moe.raw_state_dict()

    # The scales in the grouped matmul's layout, [E, N/128, K/128, 32, 4, 4].
    assert {
        name: (weight.dtype, list(weight.shape))
        for name, weight in weights.items()
    } == {
        "experts.gate_up_proj": (
            DType.uint8,
            [EXPERTS, 2 * WIDTH, HIDDEN // 2],
        ),
        "experts.gate_up_proj_scale": (
            DType.float8_e8m0fnu,
            [EXPERTS, 2 * WIDTH // 128, HIDDEN // 128, 32, 4, 4],
        ),
        "experts.down_proj": (DType.uint8, [EXPERTS, HIDDEN, WIDTH // 2]),
        "experts.down_proj_scale": (
            DType.float8_e8m0fnu,
            [EXPERTS, HIDDEN // 128, WIDTH // 128, 32, 4, 4],
        ),
        "gate.gate_score.weight": (DType.float32, [EXPERTS, HIDDEN]),
        "gate.e_score_correction_bias": (DType.float32, [EXPERTS]),
    }
