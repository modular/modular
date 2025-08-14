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
"""Library for graph shape types."""

from __future__ import annotations

import sys
from collections.abc import Iterable

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

from max._core.dialects import builtin, mosh

from .dim import Dim, DimLike, StaticDim, SymbolicDim


class Shape(list[Dim]):
    def __init__(self, dims: ShapeLike = ()) -> None:
        super().__init__(Dim(dim) for dim in dims)

    @property
    def rank(self):
        return len(self)

    def to_mlir(self) -> mosh.ShapeAttr:
        shape_type = mosh.ShapeType()
        return mosh.ShapeAttr([dim.to_mlir() for dim in self], shape_type)

    @classmethod
    def from_mlir(cls, attr: builtin.TypedAttr) -> Shape:
        if not isinstance(attr, mosh.ShapeAttr):
            raise TypeError(
                f"Shape.from_mlir only supported for mosh.ShapeAttr, got {attr}"
            )
        return cls([Dim.from_mlir(dim) for dim in attr.values])

    @property
    def static_dims(self) -> list[int]:
        """Returns all static dims in the shape as a list of integers."""
        return [d.dim for d in self if isinstance(d, StaticDim)]

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Lists the symbolic dimension names on which this shape depends."""
        for dim in self:
            yield from dim.parameters

    # TypeGuard and TypeIs don't support self/cls narrowing
    @staticmethod
    def is_static(shape: Shape) -> TypeGuard[StaticShape]:
        return all(isinstance(dim, StaticDim) and dim.dim >= 0 for dim in shape)


StaticShape = list[StaticDim]
ShapeLike = Iterable[DimLike]
