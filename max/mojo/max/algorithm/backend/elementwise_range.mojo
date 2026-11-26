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
"""Runs an elementwise body over one contiguous range of a row-major shape.

A target that splits an elementwise loop into one contiguous range per thread,
rather than striding it, runs each thread's range with `elementwise_range`.
"""

from std.utils.coord import Coord


@always_inline
def elementwise_range[
    simd_width: Int
](
    func: Some[def[width: Int, alignment: Int = 1](Coord) -> None],
    extents: Coord,
    start: Int,
    end: Int,
):
    """Calls `func` on the elements of `extents` from row-major offset `start`
    up to `end`.

    It calls `func` with `simd_width` elements of one row where it can, and
    with one element otherwise. In rank 1 a full-width call starts at a
    multiple of `simd_width`.

    Parameters:
        simd_width: The width of a full-width call.

    Args:
        func: The body, called with the index of the first element it covers.
        extents: The flat shape whose elements to run.
        start: The row-major offset of the first element to run.
        end: The row-major offset past the last element to run.
    """
    comptime rank = extents.rank
    comptime assert extents.is_flat, "extents must be flat"

    comptime if rank == 1:
        var i = start
        # Alignment prefix: scalar elements until `i` reaches the next
        # simd_width-aligned boundary. Without this, workers whose
        # `start` is not a multiple of `simd_width` would execute a bulk
        # iteration that spans two logical rows, reading the wrong gather
        # index for the overflow elements.
        while i % simd_width != 0 and i < end:
            func[1](Coord(i))
            i += 1
        while i + simd_width <= end:
            func[simd_width](Coord(i))
            i += simd_width
        while i < end:
            func[1](Coord(i))
            i += 1
    else:
        if start >= end:
            return

        var idx = Array[Int, rank](fill=0)
        var rem = start
        comptime for d_rev in range(rank - 1, 0, -1):
            idx[d_rev] = rem % Int(extents[d_rev])
            rem //= Int(extents[d_rev])
        idx[0] = rem

        var col_i = idx[rank - 1]
        var linear_i = start
        while linear_i < end:
            var row_end = min(Int(extents[rank - 1]), col_i + end - linear_i)
            linear_i += row_end - col_i

            while col_i + simd_width <= row_end:
                idx[rank - 1] = col_i
                func[simd_width](Coord(idx))
                col_i += simd_width

            while col_i < row_end:
                idx[rank - 1] = col_i
                func[1](Coord(idx))
                col_i += 1

            col_i = 0
            if linear_i < end:
                # Advance `idx` to the start of the next row.
                idx[rank - 1] = 0
                var carry = True
                comptime for d_rev in range(rank - 2, 0, -1):
                    if carry:
                        idx[d_rev] += 1
                        carry = idx[d_rev] >= Int(extents[d_rev])
                        if carry:
                            idx[d_rev] = 0
                if carry:
                    idx[0] += 1
