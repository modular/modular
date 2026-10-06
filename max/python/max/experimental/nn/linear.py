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
"""A Module for linear transformations."""

from __future__ import annotations

from typing import Literal

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.sharding import (
    DeviceMapping,
    Partial,
    Placement,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Dim, DimLike, TensorValue
from max.nn.kernels import (
    matmul_static_scaled_float8,
    quantize_static_scaled_float8,
)
from max.nn.quant_config import QuantConfig

from .module import Module, PinnedDeviceTensor


class Linear(Module[[Tensor], Tensor]):
    """A unary linear transformation over an input tensor.

    Linear is defined as `f(x) = x @ W.T + B` where `W` is the
    weight tensor and B is an optional bias tensor.

    If W is not square then the transformation represents a
    dimensionality change. By convention the weight tensor is stored
    transposed.

    .. code-block:: python

        from max.experimental.nn import Linear
        from max.experimental.tensor import Tensor

        model = Linear(5, 10)

        assert dict(model.parameters) == {
            "weight": model.weight, "bias": model.bias
        }

        result = model(Tensor.ones([5]))
        assert result.shape == [10]
    """

    # By convention weight is stored transposed
    # ie. weight.shape == [out_dim, in_dim]
    weight: Tensor
    """The weight :obj:`~max.experimental.tensor.Tensor` for the linear transformation."""
    bias: Tensor | Literal[0]
    """The bias :obj:`~max.experimental.tensor.Tensor` for the linear transformation (or 0 if bias is disabled)."""
    weight_scale: PinnedDeviceTensor
    """The per-tensor weight scale, on the CPU. Present only with a
    ``quant_config``."""
    input_scale: PinnedDeviceTensor
    """The static per-tensor input scale, on the CPU. Present only with a
    ``quant_config``."""

    def __init__(
        self,
        in_dim: DimLike,
        out_dim: DimLike,
        *,
        bias: bool = True,
        quant_config: QuantConfig | None = None,
    ):
        """Constructs a random linear transformation of the given dimensions.

        Args:
            in_dim: The dimensionality of the input to the transformation
            out_dim: The dimensionality after applying the transformation
                to the input tensor of dim `in_dim`.
            bias: Whether to use a `bias` in the transformation.
            quant_config: The :class:`~max.nn.quant_config.QuantConfig` of a
                quantized weight. Only static per-tensor FP8 is supported: an
                ``float8_e4m3fn`` weight with a ``weight_scale`` and an
                ``input_scale``, which start as zeros.

        Raises:
            NotImplementedError: If ``quant_config`` is not static per-tensor
                FP8.
        """
        self.quant_config = quant_config
        if quant_config is None:
            self.weight = random.normal([out_dim, in_dim])
        else:
            if not _is_static_fp8_per_tensor(quant_config):
                raise NotImplementedError(
                    "max.experimental.nn.Linear supports only static "
                    f"per-tensor FP8 quantization, not {quant_config}"
                )
            self.weight = Tensor.zeros(
                [out_dim, in_dim], dtype=DType.float8_e4m3fn
            )
            self.weight_scale = Tensor.zeros(
                [], dtype=quant_config.weight_scale.dtype, device=CPU()
            )
            self.input_scale = Tensor.zeros(
                [1], dtype=quant_config.input_scale.dtype, device=CPU()
            )
        self.bias = random.normal([out_dim]) if bias else 0

    @property
    def in_dim(self) -> Dim:
        """The input dimension for the transformation."""
        return self.weight.shape[1]

    @property
    def out_dim(self) -> Dim:
        """The output dimension for the transformation."""
        return self.weight.shape[0]

    def __rich_repr__(self):
        """Repr matching the Linear constructor."""
        yield "in_dim", self.in_dim
        yield "out_dim", self.out_dim
        yield "bias", isinstance(self.bias, Tensor), True

    def forward(self, x: Tensor) -> Tensor:
        """Applies a linear transformation to the input tensor.

        Linear is defined as `f(x) = x @ W.T + B` where `W` is the
        weight tensor and B is an optional bias tensor.

        Args:
            x: The input tensor
        Returns:
            The result of applying the linear transformation to the tensor.
        """
        if self.quant_config is not None:
            y = _float8_matmul_op(
                x, self.weight, self.weight_scale, self.input_scale
            )
            if y.is_distributed:
                y = y.rebind_mapping(_matmul_mapping(x, self.weight))
            return y + self.bias
        return x @ self.weight.T + self.bias


def _matmul_mapping(x: Tensor, weight: Tensor) -> DeviceMapping:
    """Returns the placement of ``x @ weight.T`` on each mesh axis.

    The FP8 matmul has no sharding rule, so it runs on each device's shards
    as placed. A weight split on its output rows splits the output columns,
    one split on its input columns leaves partial sums, and a replicated one
    keeps the activation's placement.
    """
    placements: list[Placement] = []
    for x_placement, w_placement in zip(
        x.placements, weight.placements, strict=True
    ):
        if w_placement == Sharded(0):
            placements.append(Sharded(x.rank - 1))
        elif w_placement == Sharded(1):
            placements.append(Partial())
        else:
            placements.append(x_placement)
    return DeviceMapping(x.mesh, tuple(placements))


def _is_static_fp8_per_tensor(quant_config: QuantConfig) -> bool:
    return (
        quant_config.is_static
        and quant_config.input_scale.is_tensor
        and quant_config.weight_scale.is_tensor
        and not quant_config.is_fp4
        and not quant_config.is_mx
        and not quant_config.is_int8_w8a8
    )


def _float8_matmul(
    x: Tensor, weight: Tensor, weight_scale: Tensor, input_scale: Tensor
) -> TensorValue:
    x_fp8 = quantize_static_scaled_float8(
        TensorValue(x), TensorValue(input_scale), scale_is_inverted=False
    )
    return matmul_static_scaled_float8(
        x_fp8,
        TensorValue(weight),
        TensorValue(input_scale),
        TensorValue(weight_scale),
    )


_float8_matmul_op = F.functional(_float8_matmul)
