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
"""The attention and Gated DeltaNet projections, as the checkpoint stores them.

A modelopt Qwen3.5 checkpoint keeps these projections in per-tensor FP8,
which has no quantized-tensor container; every other checkpoint keeps them
unquantized. :class:`Projection` holds one projection's tensors under the
checkpoint's names. The functions below run one device's share of one
projection, or of a stack of them as one matmul.
"""

from __future__ import annotations

from collections.abc import Sequence

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.kernels import quantize_static_scaled_float8, quantized_matmul
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

__all__ = [
    "Projection",
    "ProjectionTensors",
    "matmul",
    "stacked_matmul",
]


@tree.dataclass(frozen=True)
class ProjectionTensors:
    """One projection's tensors, as a per-device function reads them."""

    weight: Tensor
    quant_config: QuantConfig | None = None
    weight_scale: Tensor | None = None
    input_scale: Tensor | None = None


class Projection(Module[[Tensor], Tensor]):
    """One projection's weight and, when quantized, its scales.

    The weight is ``[out_dim, in_dim]`` as the checkpoint stores it. A
    per-tensor FP8 projection adds its ``weight_scale`` and ``input_scale``,
    which stay on the host.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dtype: DType,
        quant_config: QuantConfig | None = None,
        sharding: tuple[str | None, str | None] | None = None,
        on_host: bool = False,
    ) -> None:
        """Declares the projection's tensors.

        Args:
            in_dim: The input width.
            out_dim: The output width.
            dtype: The dtype of an unquantized weight.
            quant_config: The checkpoint's per-tensor FP8 quantization of
                this projection.
            sharding: The tensor-parallel spec of the weight: ``(TP, None)``
                splits the output rows and ``(None, TP)`` the inputs.
                ``None`` replicates it.
            on_host: Whether to keep the weight on the host, for a caller
                that places each device's share itself.

        Raises:
            ValueError: If ``quant_config`` isn't per-tensor FP8.
        """
        self.quant_config = quant_config
        device = CPU() if on_host else None
        if quant_config is None:
            self.weight = Tensor.zeros(
                [out_dim, in_dim], dtype=dtype, device=device
            )
        else:
            if quant_config.is_nvfp4:
                raise ValueError(
                    "Qwen3.5's attention and Gated DeltaNet projections are"
                    " per-tensor FP8 or unquantized, not NVFP4"
                )
            self.weight = Tensor.zeros(
                [out_dim, in_dim], dtype=DType.float8_e4m3fn, device=device
            )
            self.weight_scale = Tensor.zeros(
                [], dtype=DType.float32, device=CPU()
            )
            self.input_scale = Tensor.zeros(
                [], dtype=DType.float32, device=CPU()
            )
        if sharding is not None and not on_host:
            self.weight = self.weight.to(
                NamedMapping(self.weight.mesh, sharding)
            )

    @property
    def tensors(self) -> ProjectionTensors:
        """The projection's tensors, for a per-device function."""
        if self.quant_config is None:
            return ProjectionTensors(weight=self.weight)
        return ProjectionTensors(
            weight=self.weight,
            quant_config=self.quant_config,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x @ weight.T`` on one device."""
        return matmul(x, self.tensors)


def _max_scalar(scales: Sequence[Tensor | None]) -> Tensor:
    """Returns the largest of the scalar ``scales``, as a scalar."""
    values = []
    for scale in scales:
        assert scale is not None
        values.append(scale.reshape([1]))
    return F.max(F.concat(values)).reshape([])


def matmul(x: Tensor, proj: ProjectionTensors) -> Tensor:
    """Returns ``x @ proj.weight.T``, quantizing ``x`` as the weight needs."""
    if proj.quant_config is None:
        return x @ proj.weight.T
    return quantized_matmul(
        x,
        proj.weight,
        proj.weight_scale,
        proj.input_scale,
        proj.quant_config,
    )


def stacked_matmul(x: Tensor, projs: Sequence[ProjectionTensors]) -> Tensor:
    """Returns ``x`` through the projections stacked as one matmul.

    The output holds each projection's columns in order. Per-tensor FP8
    weights are each dequantized and requantized under the largest weight
    scale, and the activation is quantized under the largest input scale.
    """
    quant_config = projs[0].quant_config
    weights = [p.weight for p in projs]
    if quant_config is None:
        return x @ F.concat(weights).T
    input_scale = _max_scalar([p.input_scale for p in projs])
    dequantized = []
    for p in projs:
        assert p.weight_scale is not None
        dequantized.append(p.weight * p.weight_scale.to(p.weight.device))
    weight_scale = _max_scalar([p.weight_scale for p in projs])
    weight = quantize_static_scaled_float8(F.concat(dequantized), weight_scale)
    return quantized_matmul(x, weight, weight_scale, input_scale, quant_config)
