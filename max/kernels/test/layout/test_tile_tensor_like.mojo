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

"""Checks `_TileTensorLike`, which exposes a `TileTensor`'s parameters to
generic code."""

from std.testing import TestSuite
from layout import DefaultEngine, TileTensor, row_major
from layout.tile_tensor import _TileTensorLike

comptime _Layout = type_of(row_major[128, 1]())
comptime _Tile = TileTensor[DType.float32, _Layout, MutUntrackedOrigin]


def test_tile_tensor_is_tile_tensor_like() raises:
    comptime assert conforms_to(_Tile, _TileTensorLike)
    comptime assert not conforms_to(Int32, _TileTensorLike)


def test_tile_tensor_like_exposes_parameters() raises:
    comptime assert _Tile._dtype == DType.float32
    comptime assert TypeList.of[_Layout]().contains[_Tile._LayoutType]()
    comptime assert TypeList.of[DefaultEngine[element_width=1]]().contains[
        _Tile._Engine
    ]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
