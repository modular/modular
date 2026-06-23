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

"""Provides layer normalization for experimental tensors."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

from ..module import Module


def layer_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float,
    keep_dtype: bool,
) -> Tensor:
    """Applies layer normalization to ``x`` over its last dimension.

    When ``keep_dtype`` is ``True``, the reduction runs in the input dtype.
    When ``keep_dtype`` is ``False``, ``x``, ``gamma``, and ``beta`` are
    cast to float32, the normalization runs in float32, and the result is
    cast back to the dtype of ``x``.

    Args:
        x: The tensor to normalize. Reduction runs over the last axis.
        gamma: The scale applied after normalization. A 1-D tensor whose
            length matches the last dimension of ``x``.
        beta: The bias added after scaling. A 1-D tensor with the same
            shape as ``gamma``.
        eps: A small positive constant added to the variance for numerical
            stability.
        keep_dtype: Whether to run the reduction in the input dtype. Pass
            ``False`` to upcast to float32 for the reduction and cast back.

    Returns:
        A tensor with the same shape and dtype as ``x``.
    """
    if keep_dtype:
        return F.layer_norm(x, gamma=gamma, beta=beta, epsilon=eps)
    output = F.layer_norm(
        F.cast(x, DType.float32),
        gamma=F.cast(gamma, DType.float32),
        beta=F.cast(beta, DType.float32),
        epsilon=eps,
    )
    return F.cast(output, x.dtype)


class LayerNorm(Module[[Tensor], Tensor]):
    """Layer normalization over the last dimension of the input.

    Takes an integer ``dim`` and always reduces over the last axis. By
    default the reduction runs in the input dtype. Pass ``keep_dtype=False``
    to upcast to float32 for the reduction and cast back, which trades a
    small amount of throughput for numerical stability on float16 or
    bfloat16 inputs.

    For example:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental.nn.norm import LayerNorm
        from max.experimental.realization_context import (
            GraphRealizationContext,
            realization_context,
        )
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, Graph, TensorType

        graph = Graph(
            "ln",
            input_types=[
                TensorType(DType.float32, ("batch", "seq", 2048), DeviceRef.GPU()),
            ],
        )
        ctx = GraphRealizationContext(graph)
        with realization_context(ctx), ctx:
            x = Tensor.from_graph_value(graph.inputs[0])
            norm = LayerNorm(2048)
            y = norm(x)
            graph.output(y)

    Args:
        dim: The size of the last dimension of the input.
        eps: A small positive constant added to the variance for numerical
            stability. Defaults to ``1e-5``.
        keep_dtype: Whether to run the reduction in the input dtype. Pass
            ``False`` to upcast to float32 for the reduction and cast back.
            Defaults to ``True``.
        elementwise_affine: Whether to learn a per-element scale (and
            optional bias). When ``False``, no parameters are created and
            the normalized output is returned directly. Defaults to ``True``.
        use_bias: Whether to learn an additive bias. Only effective when
            ``elementwise_affine`` is ``True``. Defaults to ``True``.
    """

    weight: Tensor | None
    """The learned per-element scale of shape ``[dim]``, or ``None`` when
    ``elementwise_affine`` is ``False``."""

    bias: Tensor | None
    """The learned per-element bias of shape ``[dim]``, or ``None`` when
    ``elementwise_affine`` is ``False`` or ``use_bias`` is ``False``."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        keep_dtype: bool = True,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.keep_dtype = keep_dtype
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Tensor.ones([dim])
            self.bias = Tensor.zeros([dim]) if use_bias else None
        else:
            self.weight = None
            self.bias = None

    def __rich_repr__(self):
        """Yields fields for the rich debug repr."""
        yield "dim", self.dim
        yield "eps", self.eps, 1e-5

    def _affine_params(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.weight is None:
            gamma = F.broadcast_to(
                F.constant(1.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            gamma = self.weight

        if self.bias is None:
            beta = F.broadcast_to(
                F.constant(0.0, dtype=x.dtype, device=x.device),
                shape=(x.shape[-1],),
            )
        else:
            beta = self.bias

        return gamma, beta

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``x`` normalized over its last dimension."""
        gamma, beta = self._affine_params(x)
        return layer_norm(x, gamma, beta, self.eps, self.keep_dtype)
