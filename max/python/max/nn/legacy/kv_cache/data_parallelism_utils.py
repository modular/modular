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

from collections.abc import Sequence
from typing import TypeVar

from max.graph import TensorValue, ops
from max.support.math import ceildiv

T = TypeVar("T")


def split_into_groups(x: Sequence[T], groups: int) -> list[Sequence[T]]:
    """Evenly split the items into groups.

    Examples:
        >>> split_into_groups([0, 1, 2, 3, 4, 5, 6, 7], 2)
        [[0, 1, 2, 3], [4, 5, 6, 7]]

        >>> split_into_groups([0, 1, 2, 3, 4, 5, 6, 7], 3)
        [[0, 1, 2], [3, 4, 5], [6, 7]]

    Args:
        x: The list of items to split.
        groups: The number of groups to create.

    Returns:
        A list of lists of items, where each inner list contains items for
        a single group.
    """
    num_items_per_group = ceildiv(len(x), groups)
    return [
        x[i * num_items_per_group : (i + 1) * num_items_per_group]
        for i in range(groups)
    ]


def split_input_row_offsets(
    num_replicas: int,
    input_row_offsets: TensorValue,
    data_parallel_splits: TensorValue,
) -> list[TensorValue]:
    """Split the input row offsets into data parallel splits.


    Example:
        num_replicas = 2
        input_row_offsets = [0, offset_1, offset_2, offset_3, offset_4]
        data_parallel_splits = [0, 2, 4]

    Outputs:
        split_offsets = [0, offset_1, offset_2], [0, new_offset_3, new_offset_4]

    Where the new offsets are computed by subtracting the previous offset from
    the current offset (e.g. `new_offset_3 = offset_3 - offset_2`).

    Args:
        num_replicas: The number of replicas to split the input row offsets into.
        input_row_offsets: The input row offsets to split.
        data_parallel_splits: The data parallel splits to use. The size of
            data_parallel_splits must be equal to the number of replicas + 1.

    Returns:
        A list of input row offsets, one per data parallel split.
    """
    split_offsets = []
    for i in range(num_replicas):
        if i + 1 >= num_replicas:
            end_idx = None
        else:
            end_idx = data_parallel_splits[i + 1] + 1

        offsets_slice = ops.slice_tensor(
            input_row_offsets,
            [
                (
                    slice(data_parallel_splits[i], end_idx),
                    f"offset_split_{i}",
                )
            ],
        )
        offsets_slice -= input_row_offsets[data_parallel_splits[i]]
        split_offsets.append(offsets_slice)
    return split_offsets
