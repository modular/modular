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

"""Root mean square layer normalization."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.sharding import PlacementMapping
from max.experimental.sharding.rules._common import (
    is_partial,
    resolve_partials_mapping,
)
from max.experimental.tensor import Tensor
from max.graph import Dim, TensorValue, ops

from ..module import Module

_rms_norm = F.functional(ops.rms_norm)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float,
    weight_offset: float = 0.0,
    multiply_before_cast: bool = False,
) -> Tensor:
    """Applies Root Mean Square layer normalization to an input tensor.

    See https://arxiv.org/abs/1910.07467

    Args:
        x: The input tensor
        weight: The weights for the normalization
        eps: A value added to the denominator of the normalization for
            numerical stability
        weight_offset: A value added to the weights before normalization.
            Typically 1 for Gemma-like normalization and 0 otherwise.
        multiply_before_cast: Whether to multiply before or after
            casting to the output dtype. Typically True for Gemma-like
            normalization and False otherwise.

    Returns:
        A layer-normalized tensor with the same shape and type as `x`.
    """
    # TODO: Merge once we have a better way to define placement rules for custom
    # ops.
    if x.is_distributed:
        return _rms_norm_distributed(
            x, weight, eps, weight_offset, multiply_before_cast
        )
    return _rms_norm(x, weight, eps, weight_offset, multiply_before_cast)


def _resolve_partials(x: Tensor) -> Tensor:
    if x.is_distributed and any(is_partial(p) for p in x.placements):
        return x.to(resolve_partials_mapping(x.mapping))
    return x


def _rms_norm_distributed(
    x: Tensor,
    weight: Tensor,
    eps: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> Tensor:
    x = _resolve_partials(x)
    mesh = x.mesh
    x_shards = x.local_shards
    w_shards = weight.local_shards if weight.is_distributed else None

    results = []
    for i in range(mesh.num_devices):
        w_i = w_shards[i] if w_shards is not None else weight
        result = ops.rms_norm(
            x_shards[i], w_i, eps, weight_offset, multiply_before_cast
        )
        results.append(result)

    return Tensor.from_shard_values(
        tuple(TensorValue(r) for r in results),
        PlacementMapping(mesh, x.placements),
    )


class RMSNorm(Module[[Tensor], Tensor]):
    """Computes the Root Mean Square normalization on inputs."""

    weight: Tensor
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Constructs RMSNorm.

        Args:
            dim: Size of last dimension of the expected input.
            eps: Value added to denominator for numerical stability.
        """
        self.weight = Tensor.ones([dim])
        self.eps = eps

    @property
    def dim(self) -> Dim:
        """Returns the embedding dimension."""
        return self.weight.shape[0]

    def __rich_repr__(self):
        """Repr matching the Linear constructor."""
        yield "dim", self.dim
        yield "eps", self.eps, 1e-6

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMS normalization to the input."""
        return rms_norm(x, self.weight, self.eps)


class GemmaRMSNorm(RMSNorm):
    """Computes the Root Mean Square normalization on inputs.

    Differences to traditional RMSNorm:
    - x * (1 + w) instead of x * w.
    - (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies Gemma-style RMS normalization to the input."""
        return rms_norm(
            x,
            self.weight,
            self.eps,
            weight_offset=1.0,
            multiply_before_cast=True,
        )
