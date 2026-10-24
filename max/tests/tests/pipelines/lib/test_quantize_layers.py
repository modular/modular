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
from max.experimental.nn.common_layers.quant_linear import QuantizedLinear
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
