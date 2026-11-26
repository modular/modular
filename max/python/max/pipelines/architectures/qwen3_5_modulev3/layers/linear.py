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
from typing import Any

from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import FP8BlockTensor, QTensor
from max.nn.kernels import quantize_static_scaled_float8, quantized_matmul
from max.nn.module import _TransparentKey
from max.nn.quant_config import QuantConfig, QuantFormat
from max.tensor import Tensor
from max.tensor.sharding import NamedMapping

from .sharding import check_block_split

__all__ = [
    "Projection",
    "ProjectionTensors",
    "matmul",
    "stacked_matmul",
    "weight_rows",
]


@tree.dataclass(frozen=True)
class ProjectionTensors:
    """One projection's tensors, as a per-device function reads them."""

    weight: Tensor | QTensor
    quant_config: QuantConfig | None = None
    weight_scale: Tensor | None = None
    input_scale: Tensor | None = None


class Projection(Module[[Tensor], Tensor]):
    """One projection's weight and, when quantized, its scales.

    The weight is ``[out_dim, in_dim]`` as the checkpoint stores it. A
    per-tensor FP8 projection adds its ``weight_scale`` and ``input_scale``,
    which stay on the host; a block-scaled one holds an
    :class:`~max.nn.common_layers.quant_tensor.FP8BlockTensor`, which carries
    its per-block scales itself.
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
            quant_config: The checkpoint's quantization of this projection:
                per-tensor FP8, block-scaled FP8, or ``None``.
            sharding: The tensor-parallel spec of the weight: ``(TP, None)``
                splits the output rows and ``(None, TP)`` the inputs.
                ``None`` replicates it.
            on_host: Whether to keep the weight on the host, for a caller
                that places each device's share itself.

        Raises:
            ValueError: If ``quant_config`` asks for a format this projection
                does not serve.
        """
        self.quant_config = quant_config
        device = CPU() if on_host else None
        self.weight: Tensor | QTensor
        if quant_config is None:
            self.weight = Tensor.zeros(
                [out_dim, in_dim], dtype=dtype, device=device
            )
        elif quant_config.format is QuantFormat.BLOCKSCALED_FP8:
            self.weight = quant_ops.quantized_weight(
                out_dim, in_dim, quant_config
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
            if isinstance(self.weight, FP8BlockTensor):
                check_block_split(
                    [int(d) for d in self.weight.weight.shape],
                    self.weight.block_size,
                    sharding,
                    self.weight.mesh,
                )
            self.weight = self.weight.to(
                NamedMapping(self.weight.mesh, sharding)
            )

    @property
    def tensors(self) -> ProjectionTensors:
        """The projection's tensors, for a per-device function."""
        if isinstance(self.weight, QTensor):
            return ProjectionTensors(
                weight=self.weight, quant_config=self.quant_config
            )
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

    def __tree_flatten__(self) -> tuple[dict[Any, Any], None]:
        """Returns the children, with no path segment for a quantized weight.

        An :class:`FP8BlockTensor` already names its fields as the checkpoint
        names this projection's tensors, ``weight`` beside
        ``weight_scale_inv``, so it adds no segment of its own.
        """
        children, meta = super().__tree_flatten__()
        return {
            _TransparentKey(key)
            if key == "weight" and isinstance(value, QTensor)
            else key: value
            for key, value in children.items()
        }, meta


def weight_rows(weight: Tensor | QTensor) -> int:
    """Returns a projection weight's output width.

    A block-scaled weight carries it on its unpacked ``float8`` payload, so
    the container answers like the plain weight it replaces.
    """
    if isinstance(weight, FP8BlockTensor):
        return int(weight.weight.shape[0])
    if isinstance(weight, Tensor):
        return int(weight.shape[0])
    raise ValueError(
        f"{type(weight).__name__} has no output width; Qwen3.5 reads a"
        " block-scaled weight or a plain one, not another container."
    )


def _max_scalar(scales: Sequence[Tensor | None]) -> Tensor:
    """Returns the largest of the scalar ``scales``, as a scalar."""
    values = []
    for scale in scales:
        assert scale is not None
        values.append(scale.reshape([1]))
    return F.max(F.concat(values)).reshape([])


def _block_weight(weight: FP8BlockTensor) -> FP8BlockTensor:
    """Widens bf16 block scales to the float32 the block matmul requires.

    Hugging Face fine-grained FP8 checkpoints (``quant_method: "fp8"``) ship
    one ``weight_scale_inv`` per 128x128 block, as bf16 in the Qwen3.8-27B-FP8
    checkpoint; the graph op takes float32 block scales beside its float32
    activation scales. :meth:`FP8BlockTensor.zeros` declares the scales
    float32, so the pipeline's weight binding casts bf16 ones on the host
    before the graph runs and this returns the weight unchanged. The cast here
    covers a scale that reaches the graph narrower than that; it is exact.
    """
    if weight.weight_scale_inv.dtype == DType.float32:
        return weight
    return FP8BlockTensor(
        weight=weight.weight,
        weight_scale_inv=weight.weight_scale_inv.cast(DType.float32),
        block_size=weight.block_size,
    )


def _as_block(weight: Tensor | QTensor) -> FP8BlockTensor:
    """Returns ``weight`` as a block-scaled container, or refuses it.

    A projection set is quantized as one unit, so a stack holding a weight of
    another kind is a checkpoint this path cannot read.
    """
    if not isinstance(weight, FP8BlockTensor):
        raise ValueError(
            f"a block-scaled projection set holds a {type(weight).__name__}"
        )
    return weight


def _as_plain(weight: Tensor | QTensor) -> Tensor:
    """Returns ``weight`` as a plain tensor, or refuses another container."""
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"a per-tensor FP8 projection sets a plain weight, not a"
            f" {type(weight).__name__}"
        )
    return weight


def matmul(x: Tensor, proj: ProjectionTensors) -> Tensor:
    """Returns ``x @ proj.weight.T``, quantizing ``x`` as the weight needs."""
    if isinstance(proj.weight, FP8BlockTensor):
        return quant_ops.matmul(x, _block_weight(proj.weight))
    weight = _as_plain(proj.weight)
    if proj.quant_config is None:
        return x @ weight.T
    return quantized_matmul(
        x,
        weight,
        proj.weight_scale,
        proj.input_scale,
        proj.quant_config,
    )


def stacked_matmul(x: Tensor, projs: Sequence[ProjectionTensors]) -> Tensor:
    """Returns ``x`` through the projections stacked as one matmul.

    The output holds each projection's columns in order. Per-tensor FP8
    weights are each dequantized and requantized under the largest weight
    scale, and the activation is quantized under the largest input scale.
    Block-scaled weights take one matmul each: their scales are per block, so
    stacking them under one scale would change the arithmetic.
    """
    if isinstance(projs[0].weight, FP8BlockTensor):
        return F.concat(
            [
                quant_ops.matmul(x, _block_weight(_as_block(p.weight)))
                for p in projs
            ],
            axis=-1,
        )
    quant_config = projs[0].quant_config
    weights = [_as_plain(p.weight) for p in projs]
    if quant_config is None:
        return x @ F.concat(weights).T
    input_scale = _max_scalar([p.input_scale for p in projs])
    dequantized = []
    for weight, p in zip(weights, projs, strict=True):
        assert p.weight_scale is not None
        dequantized.append(weight * p.weight_scale.to(weight.device))
    weight_scale = _max_scalar([p.weight_scale for p in projs])
    weight = quantize_static_scaled_float8(F.concat(dequantized), weight_scale)
    return quantized_matmul(x, weight, weight_scale, input_scale, quant_config)
