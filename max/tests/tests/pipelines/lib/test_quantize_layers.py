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
"""Layer substitution, and the format ``--quantization-format`` resolves to."""

from __future__ import annotations

import pytest
from max._entrypoints.cli.quantize import (
    insert_quantized_linears,
    on_the_fly_config,
    quant_formats,
)
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
    QuantizedQKVLinear,
)
from max.experimental.nn.common_layers.quant_tensor import NVFP4TensorQuantizer
from max.experimental.tensor import Tensor
from max.nn.quant_config import WeightSource


class _Block(Module[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = Linear(32, 64, bias=True)
        self.k_proj = Linear(32, 64, bias=False)
        self.mlp = Linear(32, 128, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return x


def test_insert_quantized_linears_wraps_only_matching_layers() -> None:
    """Name patterns become fake-quant NVFP4 linears; other linears stay put."""
    with F.lazy():
        block = _Block()
        insert_quantized_linears(
            block, [r"q_proj", r"k_proj"], on_the_fly_config("nvfp4")
        )

        assert isinstance(block.q_proj, QuantizedLinear)
        assert isinstance(block.k_proj, QuantizedLinear)
        assert isinstance(block.mlp, Linear)
        assert not isinstance(block.mlp, QuantizedLinear)

        for layer, out_dim, has_bias in (
            (block.q_proj, 64, True),
            (block.k_proj, 64, False),
        ):
            assert layer.in_dim == 32
            assert layer.out_dim == out_dim
            assert isinstance(layer.bias, Tensor) is has_bias
            assert isinstance(layer.weight, NVFP4TensorQuantizer)
            assert layer.weight.block_size == (1, 16)
            assert list(layer.weight.shape) == [out_dim, 32]

        assert {name for name, _ in block.q_proj.parameters} == {
            "weight",
            "bias",
        }
        assert {name for name, _ in block.k_proj.parameters} == {"weight"}


@pytest.mark.parametrize("name", quant_formats())
def test_every_offered_format_builds_a_quantize_on_load_config(
    name: str,
) -> None:
    """``--quantization-format`` accepts exactly what it can deliver.

    The source is the point: a config this command hands out must say the
    bytes arrive in native precision, or the weight loader would look for
    packed data that the checkpoint does not hold.
    """
    config = on_the_fly_config(name)
    assert config.source is WeightSource.QUANTIZE
    assert config.quantizes_on_load
    assert config.format.value == name


def test_an_unknown_format_names_the_ones_that_work() -> None:
    """A typo should not surface as a KeyError from a private dict."""
    with pytest.raises(ValueError, match="unknown quantization format") as exc:
        on_the_fly_config("fp4-but-faster")
    for name in quant_formats():
        assert name in str(exc.value)


def test_insert_swaps_fused_mlp_and_qkv() -> None:
    """Gate/up and q/k/v become the quant fusers when every member matches."""
    with F.lazy():
        mlp = MLP(32, 64)
        qkv = QKVLinear(32, 32, 32)
        config = on_the_fly_config("nvfp4")
        insert_quantized_linears(
            mlp, [r"^gate_proj$", r"^up_proj$", r"^down_proj$"], config
        )
        insert_quantized_linears(
            qkv, [r"^q_proj$", r"^k_proj$", r"^v_proj$"], config
        )

        assert isinstance(mlp, QuantizedMLP)
        assert isinstance(mlp.gate_proj, QuantizedLinear)
        assert isinstance(mlp.up_proj, QuantizedLinear)
        assert isinstance(mlp.down_proj, QuantizedLinear)
        assert isinstance(qkv, QuantizedQKVLinear)
        assert isinstance(qkv.q_proj, QuantizedLinear)
        assert isinstance(qkv.k_proj, QuantizedLinear)
        assert isinstance(qkv.v_proj, QuantizedLinear)


def test_insert_leaves_down_proj_on_the_plain_mlp() -> None:
    """``down_proj`` is its own matmul, so the MLP fuser stays put."""
    with F.lazy():
        mlp = MLP(32, 64)
        insert_quantized_linears(
            mlp, [r"^down_proj$"], on_the_fly_config("nvfp4")
        )

        assert type(mlp) is MLP
        assert isinstance(mlp.down_proj, QuantizedLinear)
        assert type(mlp.gate_proj) is not QuantizedLinear
        assert type(mlp.up_proj) is not QuantizedLinear


def test_insert_verbose_prints_the_patched_module(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--verbose`` prints the module after the swap, before the count log."""
    with F.lazy():
        mlp = MLP(32, 64)
        insert_quantized_linears(
            mlp, [r"^down_proj$"], on_the_fly_config("nvfp4"), verbose=True
        )

    assert "QuantizedLinear" in capsys.readouterr().out


def test_insert_rejects_half_of_a_fused_group() -> None:
    """One side of a fused matmul cannot pick a global scale by itself."""
    with F.lazy():
        mlp = MLP(32, 64)
        qkv = QKVLinear(32, 32, 32)
        config = on_the_fly_config("nvfp4")
        with pytest.raises(ValueError, match="one matmul"):
            insert_quantized_linears(mlp, [r"^gate_proj$"], config)
        with pytest.raises(ValueError, match="one matmul"):
            insert_quantized_linears(qkv, [r"^q_proj$", r"^k_proj$"], config)
        assert type(mlp) is MLP
        assert type(qkv) is QKVLinear
