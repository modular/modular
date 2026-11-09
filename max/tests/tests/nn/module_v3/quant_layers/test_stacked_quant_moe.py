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

"""Checks for the MoE layers that bind a checkpoint's whole expert stacks."""

from __future__ import annotations

from unittest.mock import MagicMock

from max import functional as F
from max.dtype import DType
from max.nn.common_layers.quant_tensor import NVFP4Tensor
from max.nn.common_layers.stacked_quant_moe import StackedQuantizedMoE
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor, default_device

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_SEQ_LEN = 4


def _layer(quant_config: QuantConfig | None = None) -> StackedQuantizedMoE:
    return StackedQuantizedMoE(
        hidden_dim=_HIDDEN_DIM,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=2,
        moe_dim=_MOE_DIM,
        quant_config=quant_config,
    )


def test_stacked_bf16_binds_one_weight_per_projection(
    mock_accelerator: MagicMock,
) -> None:
    with F.lazy():
        layer = _layer().to(mock_accelerator())
        names = {name for name, _ in layer.parameters}
        assert names == {
            "gate.gate_score.weight",
            "experts.gate_up_proj",
            "experts.down_proj",
        }
        gate_up = layer.gate_up_proj[0]
        assert isinstance(gate_up, Tensor)
        assert list(gate_up.shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]


def test_stacked_nvfp4_takes_the_checkpoint_names_and_gate_scales(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """The fused gate/up stack keeps (gate, up) scales; the layer reads the
    gate's, one per expert."""
    with F.lazy():
        layer = _layer(nvfp4_quant_config).to(mock_accelerator())
        names = {name for name, _ in layer.parameters}
        assert names == {"gate.gate_score.weight"} | {
            f"experts.{proj}.{leaf}"
            for proj in ("gate_up_proj", "down_proj")
            for leaf in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            )
        }
        stack = layer.experts.gate_up_proj
        assert isinstance(stack, NVFP4Tensor)
        assert list(stack.weight_scale_2.shape) == [_NUM_EXPERTS, 2]
        gate_up, down = layer.gate_up_proj[0], layer.down_proj[0]
        assert isinstance(gate_up, NVFP4Tensor)
        assert isinstance(down, NVFP4Tensor)
        assert list(gate_up.weight_scale_2.shape) == [_NUM_EXPERTS]
        assert list(down.input_scale.shape) == [_NUM_EXPERTS]


def test_stacked_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    device = mock_accelerator()
    # Built under the device, as a model builds it.
    with F.lazy(), default_device(device):
        layer = _layer(nvfp4_quant_config)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)
        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
