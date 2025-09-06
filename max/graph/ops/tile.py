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
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = repeats[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.
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

    output_dims = [dim * count for dim, count in zip(shape, repeats)]

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
