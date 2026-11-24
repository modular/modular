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
"""Subpackage exposing `std.utils.coord` as `layout.coord` for kernel imports."""

from std.utils.coord import (
    _CoordReplaceAt,
    ComptimeInt,
    Coord,
    CoordLike,
    DynamicCoord,
    Idx,
    coord,
    coord_to_index_list,
    crd2idx,
    idx2crd,
    _IsNotTuplePredicate,
    _CeilDiv,
    _CoordToDynamic,
    _Divide,
    _Flattened,
    _IntToComptimeInt,
    _Multiply,
    _MultiplyByScalar,
    _NestedDynamicCoord,
)

comptime UNKNOWN_VALUE = -1
"""Sentinel extent for a dimension whose size is not known at compile time.

Typed layouts model such dimensions as runtime `Scalar` values; this constant is
only for the places that build a `Coord` of extents and need a marker they can
test against, such as TMA descriptors with a dynamic leading dimension. It is
negative so that any real extent compares greater than it.
"""
