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
"""Op implementation for flatten."""

from ..value import TensorValue, TensorValueLike


def flatten(
    x: TensorValueLike, start_dim: int = 0, end_dim: int = -1
) -> TensorValue:
    """Flattens the specified dims of a symbolic tensor.

    The number and order of the elements in the tensor is unchanged.
    All dimensions from ``start_dim`` to ``end_dim`` (inclusive) are merged
    into a single output dim.

    Args:
        x: The input symbolic tensor to flatten.
        start_dim: The first dimension to flatten. Supports negative indexing.
            Defaults to 0.
        end_dim: The last dimension to flatten (inclusive). Supports negative
            indexing. Defaults to -1.

    Returns:
        A symbolic tensor with the same elements as the input, but with
        dimensions ``start_dim`` through ``end_dim`` merged into one.

    Raises:
        IndexError: If ``start_dim`` or ``end_dim`` are out of range.
        ValueError: If ``start_dim`` comes after ``end_dim``.
    """
    x = TensorValue(x)
    shape = x.shape
    rank = len(shape)
    # It is actually ok to flatten scalars. They will just become 1d tensors.
    if rank == 0:
        rank = 1

    if not (-rank <= start_dim < rank):
        raise IndexError(
            f"start_dim must be be between {-rank} and {rank - 1} (inclusive),"
            f" but was {start_dim}"
        )
    if not (-rank <= end_dim < rank):
        raise IndexError(
            f"end_dim must be be between {-rank} and {rank - 1} (inclusive), but"
            f" was {end_dim}"
        )

    end_dim = end_dim if end_dim >= 0 else end_dim + rank
    start_dim = start_dim if start_dim >= 0 else start_dim + rank

    if start_dim > end_dim:
        raise ValueError("start_dim cannot come after end_dim")

    output_shape = shape[:start_dim] + [-1] + shape[end_dim + 1 :]

    return x.reshape(output_shape)
