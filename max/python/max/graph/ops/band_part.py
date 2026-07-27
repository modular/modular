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
"""Op implementation for band_part."""

from __future__ import annotations

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import StaticDim
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def band_part(
    x: TensorValueLike,
    num_lower: int | None = None,
    num_upper: int | None = None,
    exclude: bool = False,
) -> TensorValue:
    """Masks out everything except a diagonal band of an input matrix.

    Copies the tensor, setting everything outside the central diagonal band of
    each matrix to zero. All but the last two axes are treated as batches, and
    the last two axes define the matrices.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("band_part") as graph:
            x = ops.constant(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                DType.float32,
                device=device,
            )
            # Keep the main diagonal and one sub-diagonal, producing
            # [[1, 0, 0], [1, 1, 0], [0, 1, 1]].
            graph.output(ops.band_part(x, num_lower=1, num_upper=0))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    Args:
        x: The input tensor to mask.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If ``None`` or ``-1``, includes the entire lower
            triangle. Defaults to ``None``.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If ``None`` or ``-1``, includes the entire upper
            triangle. Defaults to ``None``.
        exclude: Whether to invert the selection, zeroing out the elements in
            the band instead. Defaults to ``False``.

    Returns:
        A ``TensorValue`` representing ``x`` with the masked-out elements set to
        zero and the remaining elements copied from ``x``. It has the same shape
        and dtype as ``x``.

    Raises:
        ValueError: If the input tensor rank is less than 2, or if ``num_lower``
            or ``num_upper`` are out of bounds for statically known dimensions.
    """
    x = TensorValue(x)
    num_lower = -1 if num_lower is None else num_lower
    num_upper = -1 if num_upper is None else num_upper

    if num_lower < -1:
        raise ValueError(f"{num_lower=} must be non-negative")
    if num_upper < -1:
        raise ValueError(f"{num_upper=} must be non-negative")

    if x.rank < 2:
        raise ValueError(
            f"Input tensor {x.shape=} must have at least 2 dimensions"
        )

    # Check for out-of-bounds values for known static dimensions.
    # - m is the "vertical" dimension, and n is the "horizontal" dimension, visually
    # - num_lower is how far "down", so it is compared against m
    # - num_upper is how far "right", so it is compared against n
    *_, m, n = x.shape
    if isinstance(m, StaticDim) and num_lower >= int(m):
        raise ValueError(
            f"{num_lower=} is out of bounds for dimension size {int(m)}"
        )
    if isinstance(n, StaticDim) and num_upper >= int(n):
        raise ValueError(
            f"{num_upper=} is out of bounds for dimension size {int(n)}"
        )

    return Graph.current._add_op_generated(
        rmo.MoLinalgBandPartOp,
        x.type,
        x,
        constant(num_lower, DType.int64, DeviceRef.CPU()),
        constant(num_upper, DType.int64, DeviceRef.CPU()),
        constant(exclude, DType.bool, DeviceRef.CPU()),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
