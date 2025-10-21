# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""A Module for vector embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from ...experimental import functional as F
from ...experimental import random
from ...experimental.tensor import Tensor
from ...graph import Dim, DimLike, ShapeLike
from .module import Module


class Embedding(Module):
    """A vector embedding.

    An embedding can be thought of as a lookup table for vectors by index.
    Given an input tensor of indices into the embedding, the result
    of the embedding lookup is a tensor of the same shape, but with each index
    replaced by the value of the vector in that location in the embedding table.

    The common case for embeddings is a 1-dimensional embedding:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental.tensor import Tensor
        from max.nn.module_v3 import Embedding

        embedding = Embedding(vocab_size=1000, dim=128)
        tokens = Tensor.ones([10], dtype=DType.uint64)
        embedded = embedding(tokens)
        assert embedded.shape == [10, 128]

    However they just as easily support multi-dimensional embeddings:

    .. code-block:: python

        from max.dtype import DType
        from max.experimental.tensor import Tensor
        from max.nn.module_v3 import Embedding

        embedding = Embedding(vocab_size=1000, dims=[16, 128])
        tokens = Tensor.ones([10], dtype=DType.uint64)
        embedded = embedding(tokens)
        assert embedded.shape == [10, 16, 128]

    """

    weight: Tensor

    def __init__(
        self,
        vocab_size: DimLike,
        *,
        dim: DimLike | None = None,
        dims: ShapeLike | None = None,
    ):
        """Creates a randomly initialized embedding of the specified size.

        Args:
            vocab_size: The number of elements in the lookup table.
                Indices outside the range of [0, index_size) are illegal
                in the resulting embedding operation.
            dim: The embedding dimension if there is exactly one.
                Equivalent to `dims=[dim]`.
            dims: For specifying multi-dimensional embeddings.
                The shape of the vectors in the embedding.
        """
        if not (dim is None) ^ (dims is None):
            raise TypeError("Must specify exactly one of `dim` or `dims`")
        dims = [cast(DimLike, dim)] if dims is None else dims
        self.weight = random.normal([vocab_size, *dims])

    @property
    def vocab_size(self) -> Dim:
        """The vocab size of the embedding.

        Indices outside the range of [0, index_size) are illegal.
        """
        return self.weight.shape[0]

    @property
    def dim(self) -> Dim:
        """The dimension of the vectors in the embedding (for a 1d embedding).

        Raises: For 0- or >1-dimensional embeddings.
        """
        if self.weight.rank != 2:
            raise TypeError("Multi-dimensional embeddings must use `dims`.")
        return self.weight.shape[1]

    @property
    def dims(self) -> Sequence[Dim]:
        """The dimensions of the vectors in the embedding."""
        return self.weight.shape[1:]

    def __rich_repr__(self):
        yield "vocab_size", self.vocab_size
        if self.weight.rank == 2:
            yield "dim", self.dim
        else:
            yield "dims", self.dims

    def __call__(self, indices: Tensor) -> Tensor:
        """Applies the vector embedding to the input tensor of indices.

        Args:
            indices: An integer-valued tensor. Values must be in the range
                [0, vocab_size) for the embedding.
        Returns:
            A dense tensor made by looking up each index in the vector embedding.
            For an input of shape ``(*batch, indices)`` and an embedding of shape
            ``(vocab_size, *dims)``, the result will have shape ``(*batch, indices, *dims)``.
        """
        return F.gather(self.weight, indices, axis=0)
