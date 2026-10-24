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
"""Tests that MiMo-V2's MoE declares the checkpoint's expert tensors."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.moe import NvMxf4f8Strategy, moe_fp8
from max.pipelines.architectures.mimo_v2.layers.moe import mimo_v2_moe
from max.pipelines.architectures.mimo_v2.quant import parse_quant_scheme
from transformers.configuration_utils import PretrainedConfig

EXPERTS, HIDDEN, WIDTH = 2, 256, 512


def _config() -> PretrainedConfig:
    return PretrainedConfig(
        num_hidden_layers=2,
        moe_layer_freq=[0, 1],
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
        },
    )


def test_declares_the_checkpoint_experts() -> None:
    moe = mimo_v2_moe(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_dim=WIDTH,
        norm_topk_prob=True,
        quant_config=parse_quant_scheme(_config()).experts,
        devices=[DeviceRef.CPU()],
    )

    expected = {
        "gate.gate_score.weight": (DType.bfloat16, [EXPERTS, HIDDEN]),
        "gate.e_score_correction_bias": (DType.float32, [EXPERTS]),
    }
    for e in range(EXPERTS):
        for proj, (n, k) in (
            ("gate_proj", (WIDTH, HIDDEN)),
            ("up_proj", (WIDTH, HIDDEN)),
            ("down_proj", (HIDDEN, WIDTH)),
        ):
            base = f"experts.{e}.{proj}"
            expected[f"{base}.weight"] = (DType.uint8, [n, k // 2])
            expected[f"{base}.weight_scale"] = (
                DType.float8_e8m0fnu,
                [n, k // 32],
            )
    assert {
        name: (weight.dtype, list(weight.shape))
        for name, weight in moe.raw_state_dict().items()
    } == expected


def test_experts_quantize_inputs_to_mxfp8_on_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(moe_fp8, "accelerator_api", lambda: "cuda")
    moe = mimo_v2_moe(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_dim=WIDTH,
        norm_topk_prob=True,
        quant_config=parse_quant_scheme(_config()).experts,
        devices=[DeviceRef.CPU()],
    )
    strategy = moe._strategy()
    assert isinstance(strategy, NvMxf4f8Strategy)
    assert strategy.dtype == DType.float8_e4m3fn
