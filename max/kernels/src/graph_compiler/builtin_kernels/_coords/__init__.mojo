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

"""Comptime coordinate parsing for registrations that carry a layout string."""

from layout.coord import DynamicCoord


@inline(.always)
def _unsafe_str_to_coord[
    str_slice: StaticString
]() -> DynamicCoord[.int64, len(str_slice.split("_"))]:
    """
    Convert a string of integers separated by "_" to a coordinate.

    Parameters:
        str_slice: The string of integers separated by "_".

    Returns:
        The coordinate.
    """
    comptime size = len(str_slice.split("_"))
    var coord = DynamicCoord[.int64, size]()

    comptime for i in range(size):
        comptime sub_string = str_slice.split("_")[i]
        comptime str_len = sub_string.byte_length()
        var result = 0

        comptime for pos in range(str_len):
            result = result * 10 + (Int(sub_string.as_bytes()[pos]) - ord("0"))
        coord[i] = rebind[coord.element_types[i]](Int64(result))

    return coord
