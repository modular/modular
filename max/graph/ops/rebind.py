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
"""Op implementation for rebind."""

from __future__ import annotations

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import FilterLayout, Shape, ShapeLike
from ..value import TensorType, TensorValue, TensorValueLike


def rebind(
    x: TensorValueLike,
    shape: ShapeLike,
    message: str = "",
    layout: FilterLayout | None = None,
) -> TensorValue:
    """Rebinds a symbolic tensor to a specified set of dimensions.

    This does not mutate the symbolic tensor passed in, but instead adds a
    runtime assert that the input symbolic shape is equivalent to
    :code:`out_dims` shape. For example, if the input tensor shape has
    dynamic/unknown sizes, this will assert a fixed sizes that may be required
    for a subsequent operation.

    Args:
        x: The input symbolic tensor to rebind.
        shape: The symbolic shape to assert for ``x``, as a list of
               [``Dim``](/max/api/python/graph/type/Dim) values.
        message: The message printed if the rebind fails at runtime.
        layout: A layout of the weights used by some operations like `conv`.

    Returns:
        A symbolic tensor with the same elements and shape as the given tensor,
        but with the symbolic shape asserted to ``out_dims``.

    """
    # TODO(MSDK-662): Add checks to ensure that statically known dims are
    # rebound in a way to keep the size the same.
    v = TensorValue(x)
    shape = Shape(shape)

    if shape.rank != v.rank:
        raise ValueError(f"Wrong rank for rebind: {v.shape=} but {shape=}")

    out_type = TensorType(v.dtype, shape, device=v.device)
    out_type._layout = layout

    message_attr = mlir.StringAttr.get(message)
    return Graph.current._add_op(
        rmo.rebind_tensor_shape, out_type.to_mlir(), v, message=message_attr
    )[0].tensor
