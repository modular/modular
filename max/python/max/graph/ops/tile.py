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
from .transfer_to import transfer_to
from .validation import _check_device_placement


def tile(x: TensorValueLike, repeats: Iterable[DimLike]) -> TensorValue:
    """Returns a new tensor by tiling the input along each dimension.

    The input is copied ``N_i`` times on the i-th dimension, where
    ``N_i = repeats[i]``. The i-th dimension of the output shape is the
    i-th dimension of the input shape multiplied by ``N_i``.

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
            ``x``, or if any repeat value is not positive. Also raised for
            GPU inputs when ``strict_device_placement=DevicePlacementPolicy.Error``.
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

    old_device = x.device if not x.device.is_cpu() else None
    if old_device is not None:
        _check_device_placement("ops.tile", "TODO(GEX-2056).")
        x = transfer_to(x, DeviceRef.CPU())
    # TODO(GEX-2056): Add GPU kernel support for tile.
    result = Graph.current._add_op_generated(
        rmo.MoTileOp,
        TensorType(dtype=x.dtype, shape=output_dims, device=x.device),
        x,
        TensorValue(Shape(repeats)),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
    if old_device is not None:
        return transfer_to(result, old_device)
    return result
