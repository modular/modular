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
from layout import IntTuple, Layout, print_layout


fn row_and_column_major():
    print("row major and column major")
    var l2x4row_major = Layout.row_major(2, 4)
    print_layout(l2x4row_major)
    print()
    var l6x6col_major = Layout.col_major(6, 6)
    print_layout(l6x6col_major)
    print()


fn coords_to_index():
    print("coordinates to index")
    var l3x4row_major = Layout.row_major(3, 4)
    print_layout(l3x4row_major)

    var coords = IntTuple(1, 1)
    var idx = l3x4row_major(coords)
    print("index at (1, 1): ", idx)
    print("coordinates at index 7:", l3x4row_major.idx2crd(7))
    print()


fn nested_modes():
    print("nested modes")
    var layout_a = Layout(IntTuple(4, 4), IntTuple(4, 1))
    print_layout(layout_a)
    print()
    var layout_b = Layout(
        IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
        IntTuple(IntTuple(1, 4), IntTuple(2, 8)),
    )
    print_layout(layout_b)
    print()


def main():
    row_and_column_major()
    coords_to_index()
    nested_modes()
