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

T = TypeVar("T")


def flatten2d(list2d: Sequence[Sequence[T]]) -> list[T]:
    """Flatten a list of lists into a single list.

    Args:
        list2d: A sequence of sequences to flatten.

    Returns:
        A list containing all the flattened items in order.
    """
    return [item for sublist in list2d for item in sublist]
