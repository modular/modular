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
"""Op implementation for tile."""

from collections.abc import Iterable

from max._core.dialects import kgen, rmo

from .. import dtype_promotion
from ..dim import Dim, DimLike, StaticDim
from ..graph import Graph
from ..shape import Shape
from ..type import DeviceRef, TensorType
from ..value import TensorValue, TensorValueLike


def tile(x: TensorValueLike, repeats: Iterable[DimLike]) -> TensorValue:
    """Returns a new tensor by tiling the input along each dimension.

    The input is copied ``N_i`` times on the i-th dimension, where
    ``N_i = repeats[i]``. The i-th dimension of the output shape is the
    i-th dimension of the input shape multiplied by ``N_i``.

    .. warning::

        This operation is CPU-only. When used in a GPU-compiled graph, the
        runtime will silently transfer the tensor to CPU, execute the tile,
        then transfer the result back to GPU. In performance-sensitive paths
        (for example, inside a repeated attention layer), prefer
        :func:`concat` or :func:`broadcast_to` as GPU-native alternatives.

    Args:
        x: The input symbolic tensor to tile.
        repeats: An iterable of repeat counts, one per dimension of ``x``.
            All values must be positive. The length must equal the rank of
            ``x``.

    Returns:
        A symbolic tensor whose i-th dimension size equals
        ``x.shape[i] * repeats[i]``.

    Raises:
        ValueError: If the length of ``repeats`` does not match the rank of
            ``x``, or if any repeat value is not positive.
    """
    x = dtype_promotion._restrict_to_strong_dtypes(x)
    shape = x.shape

    repeats = list(Dim(d) for d in repeats)
    if len(shape) != len(repeats):
        raise ValueError(
            "Input rank and number of elements in repeats must match:"
            f" {shape=}, {repeats=}"
        )

    if any(count.dim <= 0 for count in repeats if isinstance(count, StaticDim)):
        raise ValueError(f"Repeats must all be positive: {repeats=}")

    output_dims = [
        dim * count for dim, count in zip(shape, repeats, strict=True)
    ]

    # TODO(GEX-2056): Add support for GPU kernel for tile and remove manual transfers
    original_device = x.type.device
    x = x.to(DeviceRef.CPU())
    answer = Graph.current._add_op_generated(
        rmo.MoTileOp,
        TensorType(dtype=x.dtype, shape=output_dims, device=x.device),
        x,
        TensorValue(Shape(repeats)),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    return answer.to(original_device)
