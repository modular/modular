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
"""Implements the built-in `swap` function.

These are Mojo built-ins, so you don't need to import them.
"""


@inline(.always)
def swap[T: Movable](mut lhs: T, mut rhs: T):
    """Swaps the values of the two arguments.

    The values are exchanged with moves, so no copies are made.

    Parameters:
        T: The type of the values. Must conform to `Movable`.

    Args:
        lhs: The value swapped with `rhs`.
        rhs: The value swapped with `lhs`.

    For example:

    ```mojo
    var a = 1
    var b = 2
    swap(a, b)
    print(a)  # 2
    print(b)  # 1
    ```
    """
    var tmp = lhs^
    lhs = rhs^
    rhs = tmp^
