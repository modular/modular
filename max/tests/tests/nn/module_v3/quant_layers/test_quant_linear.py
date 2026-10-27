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

"""Lazy-trace tests for QuantizedLinear / QuantizedMLP."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
)
from max.experimental.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
    MXFP8Quantizer,
    MXFP8Tensor,
    NVFP4Tensor,
    NVFP4TensorQuantizer,
)
from max.experimental.tensor import Tensor, default_dtype
from max.graph import Graph
from max.nn.quant_config import QuantConfig

# Dimensions are multiples of the 128x128 weight block so the FP8 scale grid
# divides evenly and stays easy to reason about.
_HIDDEN_DIM = 256
_FFN_DIM = 512
_SEQ_LEN = 4


# --------------------------------------------------------------------------- #
# QuantizedLinear
# --------------------------------------------------------------------------- #


def test_linear_bf16_parameters(mock_accelerator: MagicMock) -> None:
    """Without a quant config the weight is a single bf16 tensor."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(_HIDDEN_DIM, _FFN_DIM).to(device)

        assert isinstance(layer.weight, Tensor)
        assert list(layer.weight.shape) == [_FFN_DIM, _HIDDEN_DIM]

        names = {name for name, _ in layer.parameters}
        assert names == {"weight", "bias"}


def test_linear_bf16_no_bias_parameters(mock_accelerator: MagicMock) -> None:
    """``bias=False`` drops the bias parameter entirely."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(_HIDDEN_DIM, _FFN_DIM, bias=False).to(device)
        assert layer.bias == 0
        names = {name for name, _ in layer.parameters}
        assert names == {"weight"}


def test_linear_fp8_parameters(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """With an FP8 config the weight becomes an FP8BlockTensor."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=fp8_quant_config
        ).to(device)

        assert isinstance(layer.weight, FP8BlockTensor)
        assert (
            layer.weight.block_size == fp8_quant_config.weight_scale.block_size
        )
        assert layer.weight.weight.dtype == DType.float8_e4m3fn
        assert layer.weight.weight_scale_inv.dtype == DType.float32
        assert list(layer.weight.weight.shape) == [_FFN_DIM, _HIDDEN_DIM]
        # 512 / 128 == 4, 256 / 128 == 2
        assert list(layer.weight.weight_scale_inv.shape) == [4, 2]

        names = {name for name, _ in layer.parameters}
        assert names == {"weight", "weight_scale_inv", "bias"}


def test_linear_bf16_forward(mock_accelerator: MagicMock) -> None:
    """bf16 forward maps ``[S, in]`` -> ``[S, out]`` keeping dtype."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(_HIDDEN_DIM, _FFN_DIM).to(device)
        # The no-quant path stores a plain bf16 Tensor weight.
        weight = layer.weight
        assert isinstance(weight, Tensor)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=weight.dtype, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
        assert out.dtype == weight.dtype


def test_linear_fp8_forward(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 block-scaled forward returns a bf16 ``[S, out]`` activation."""
    device = mock_accelerator()
    with F.lazy():
        with default_dtype(DType.bfloat16):
            layer = QuantizedLinear(
                _HIDDEN_DIM, _FFN_DIM, quant_config=fp8_quant_config
            ).to(device)
            x = Tensor.zeros(
                [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
            )
            out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
        assert out.dtype == DType.bfloat16


# --------------------------------------------------------------------------- #
# QuantizedMLP
# --------------------------------------------------------------------------- #


def test_mlp_bf16_parameters(mock_accelerator: MagicMock) -> None:
    """The gated MLP has three bias-free bf16 projections by default."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMLP(_HIDDEN_DIM, _FFN_DIM).to(device)

        names = {name for name, _ in layer.parameters}
        assert names == {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }


def test_mlp_fp8_parameters(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """Each FP8 projection contributes a ``data`` + ``weight_scale_inv`` pair."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMLP(
            _HIDDEN_DIM, _FFN_DIM, quant_config=fp8_quant_config
        ).to(device)

        names = {name for name, _ in layer.parameters}
        assert names == {
            "gate_proj.weight",
            "gate_proj.weight_scale_inv",
            "up_proj.weight",
            "up_proj.weight_scale_inv",
            "down_proj.weight",
            "down_proj.weight_scale_inv",
        }


def test_mlp_bf16_forward(mock_accelerator: MagicMock) -> None:
    """bf16 MLP forward preserves the ``[S, hidden]`` shape and dtype."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMLP(_HIDDEN_DIM, _FFN_DIM).to(device)
        # The no-quant path stores plain bf16 Tensor projection weights.
        weight = layer.gate_proj.weight
        assert isinstance(weight, Tensor)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM],
            dtype=weight.dtype,
            device=device,
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == weight.dtype


def test_mlp_fp8_forward(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    """FP8 MLP forward returns a bf16 ``[S, hidden]`` activation."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMLP(
            _HIDDEN_DIM, _FFN_DIM, quant_config=fp8_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
        assert out.dtype == DType.bfloat16


# --------------------------------------------------------------------------- #
# NVFP4
# --------------------------------------------------------------------------- #


def test_linear_nvfp4_parameters(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """With an NVFP4 config the weight becomes a four-leaf NVFP4Tensor."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_quant_config
        ).to(device)

        assert isinstance(layer.weight, NVFP4Tensor)
        assert (
            layer.weight.block_size
            == nvfp4_quant_config.weight_scale.block_size
        )
        assert layer.weight.weight.dtype == DType.uint8
        assert layer.weight.weight_scale.dtype == DType.float8_e4m3fn
        # Two FP4 values per byte along K; one scale per 16-element block.
        assert list(layer.weight.weight.shape) == [_FFN_DIM, _HIDDEN_DIM // 2]
        assert list(layer.weight.weight_scale.shape) == [
            _FFN_DIM,
            _HIDDEN_DIM // 16,
        ]

        names = {name for name, _ in layer.parameters}
        assert names == {
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale",
            "bias",
        }


def test_linear_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 block-scaled forward returns a bf16 ``[S, out]`` activation."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
    assert out.dtype == DType.bfloat16


def test_linear_nvfp4_fake_quant_parameters(
    mock_accelerator: MagicMock, nvfp4_fake_quant_config: QuantConfig
) -> None:
    """Fake-quant keeps the bf16 weight as the only weight parameter."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_fake_quant_config
        ).to(device)

        assert isinstance(layer.weight, NVFP4TensorQuantizer)
        assert list(layer.weight.shape) == [_FFN_DIM, _HIDDEN_DIM]
        assert layer.weight._tensor.dtype == DType.bfloat16
        assert {name for name, _ in layer.parameters} == {"weight", "bias"}


def test_linear_nvfp4_fake_quant_forward(
    mock_accelerator: MagicMock,
    nvfp4_fake_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """On-the-fly NVFP4 quantization still returns a bf16 ``[S, out]`` activation."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_fake_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
    assert out.dtype == DType.bfloat16


def _custom_signatures(ir: str, symbol: str) -> list[tuple[str, str]]:
    """Input and output type lists for each ``mo.custom`` of ``symbol``."""
    pattern = re.compile(
        rf'symbol = "{re.escape(symbol)}".*?: \(([^)]*)\) -> (\([^)]*\)|\S+)'
    )
    return pattern.findall(ir)


def test_linear_nvfp4_fake_quant_forward_uses_packed_weight(
    mock_accelerator: MagicMock,
    nvfp4_fake_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """Forward multiplies the packed NVFP4 weight and its scales.

    The stored parameter stays bf16. It is an input of the weight quantize
    op, and the matmul consumes that op's uint8 data plus the float8 scales.
    A second forward reuses the quantization rather than quantizing again.
    """
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_fake_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)
        layer(x)
        ir = str(Graph.current._mlir_op)

    assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
    assert out.dtype == DType.bfloat16
    assert "rmo.matmul" not in ir

    bf16_weight = f"!mo.tensor<[{_FFN_DIM}, {_HIDDEN_DIM}], bf16"
    packed_weight = f"!mo.tensor<[{_FFN_DIM}, {_HIDDEN_DIM // 2}], ui8"
    matmuls = _custom_signatures(ir, "mo.matmul.dynamic.block.scaled")
    assert len(matmuls) == 2
    matmul_operands, _ = matmuls[0]
    assert packed_weight in matmul_operands
    assert bf16_weight not in matmul_operands
    assert matmul_operands.count("f8e4m3fn") == 2

    weight_quant = [
        (inputs, outputs)
        for inputs, outputs in _custom_signatures(
            ir, "mo.quantize.dynamic.block.scaled"
        )
        if bf16_weight in inputs
    ]
    assert len(weight_quant) == 1
    quant_inputs, quant_outputs = weight_quant[0]
    assert packed_weight in quant_outputs
    assert "f8e4m3fn" in quant_outputs
    assert bf16_weight in quant_inputs


def test_mlp_nvfp4_parameters(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """Each NVFP4 projection contributes its four scale/data leaves."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedMLP(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_quant_config
        ).to(device)

        names = {name for name, _ in layer.parameters}
        assert names == {
            f"{proj}.{leaf}"
            for proj in ("gate_proj", "up_proj", "down_proj")
            for leaf in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            )
        }


def test_mlp_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 MLP forward returns a bf16 ``[S, hidden]`` activation."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedMLP(
            _HIDDEN_DIM, _FFN_DIM, quant_config=nvfp4_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.dtype == DType.bfloat16


# --------------------------------------------------------------------------- #
# MXFP8
# --------------------------------------------------------------------------- #


def test_linear_mxfp8_parameters(
    mock_accelerator: MagicMock, mxfp8_quant_config: QuantConfig
) -> None:
    """With an MXFP8 config the weight is a two-leaf MXFP8Tensor."""
    device = mock_accelerator()
    with F.lazy():
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=mxfp8_quant_config
        ).to(device)

        assert isinstance(layer.weight, MXFP8Tensor)
        assert layer.weight.weight.dtype == DType.float8_e4m3fn
        assert layer.weight.weight_scale.dtype == DType.float8_e8m0fnu
        assert list(layer.weight.weight.shape) == [_FFN_DIM, _HIDDEN_DIM]
        assert list(layer.weight.weight_scale.shape) == [
            _FFN_DIM,
            _HIDDEN_DIM // 32,
        ]
        names = {name for name, _ in layer.parameters}
        assert names == {"weight", "weight_scale", "bias"}


def test_linear_mxfp8_forward(
    mock_accelerator: MagicMock,
    mxfp8_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """MXFP8 block-scaled forward returns a bf16 ``[S, out]`` activation."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=mxfp8_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
    assert out.dtype == DType.bfloat16


def test_mlp_mxfp8_forward(
    mock_accelerator: MagicMock,
    mxfp8_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """The fused gate/up MXFP8 matmul and the down matmul keep ``[S, hidden]``."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedMLP(
            _HIDDEN_DIM, _FFN_DIM, quant_config=mxfp8_quant_config
        ).to(device)
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.dtype == DType.bfloat16


def test_linear_mxfp8_fake_quant_forward(
    mock_accelerator: MagicMock,
    mxfp8_fake_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """On-the-fly MXFP8 keeps a bf16 parameter and runs the MXFP8 matmul."""
    device = mock_accelerator()
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = QuantizedLinear(
            _HIDDEN_DIM, _FFN_DIM, quant_config=mxfp8_fake_quant_config
        ).to(device)
        assert isinstance(layer.weight, MXFP8Quantizer)
        assert {name for name, _ in layer.parameters} == {"weight", "bias"}
        x = Tensor.zeros(
            [_SEQ_LEN, _HIDDEN_DIM], dtype=DType.bfloat16, device=device
        )
        out = layer(x)

        packed = layer.weight.quant_tensor
        assert isinstance(packed, MXFP8Tensor)
        assert packed.weight.dtype == DType.float8_e4m3fn
        # Normalized to the checkpoint's rank-2 layout whatever the kernel
        # returned.
        assert list(packed.weight_scale.shape) == [
            _FFN_DIM,
            _HIDDEN_DIM // 32,
        ]

    assert list(out.shape) == [_SEQ_LEN, _FFN_DIM]
    assert out.dtype == DType.bfloat16
