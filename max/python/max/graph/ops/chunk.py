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
"""Op implementation for chunk."""

from ..value import TensorValue, TensorValueLike
from .slice_tensor import slice_tensor


def chunk(x: TensorValueLike, chunks: int, axis: int = 0) -> list[TensorValue]:
    """
    Chunk the tensor into an exact number of chunks along the specified dim.

    Example:
        >>> a = TensorValue([1, 2, 3, 4, 5])
        >>> chunk(a, 2, 0)
        [TensorValue([1, 2]), TensorValue([3, 4])]

    Args:
        x: The tensor to chunk.
        chunks: The number of chunks to split the tensor into.
            `chunks` must statically evenly divide `x.shape[axis]`.
        axis: The axis to split the tensor along.

    Returns:
        A list of `chunks` tensors.
    """
    # TODO(GEX-1943): once we have control flow in the graph, this can be updated to
    # dynamic chunk counts while still supporting algebraic dims. For now,
    # this will generate exactly chunks or fail.
    x = TensorValue(x)

    if x.rank == 0 and chunks > 1:
        raise ValueError(f"Cannot split scalar value into {chunks=}")

    if axis < 0:
        axis = x.rank + axis

    if not 0 <= axis < x.rank:
        raise ValueError(f"'{axis=}' out of bounds for tensor {x=}")

    # Convert to a python bigint for int math
    n = int(x.shape[axis])

    if n % chunks != 0:
        raise ValueError(
            f"chunk: {chunks=} must statically divide {x.shape[axis]=}"
        )

    # Determine chunk size using ceiling division.
    chunk_size = n // chunks

    def slices(offset: int) -> list[slice]:
        slices = [slice(None)] * x.rank
        slices[axis] = slice(chunk_size * offset, chunk_size * (offset + 1))
        return slices

    return [slice_tensor(x, slices(offset)) for offset in range(chunks)]
