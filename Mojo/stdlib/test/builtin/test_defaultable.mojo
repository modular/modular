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

from test_utils.reflection import SimplePoint, NestedStruct
from std.testing import TestSuite, assert_equal


struct DefaultableNonMovable(Defaultable, not Movable):
    """Defines a `Defaultable` type that is not `Movable`."""

    var value: Int

    def __init__(out self):
        self.value = 5


struct HasNonMovableField(Defaultable, not Movable):
    """Derives its default `__init__()` from a non-`Movable` field."""

    var field: DefaultableNonMovable


struct DefaultableEmpty(Defaultable):
    """Derives a default `__init__()` without declaring any fields."""

    pass


struct HasArrayField(Defaultable):
    """Derives its default `__init__()` from an `Array` field."""

    var array: Array[SimplePoint, 2]


@fieldwise_init
struct NotDefaultable:
    """Defines a type without a `Defaultable` conformance."""

    var value: Int


struct Overridden(Defaultable):
    """Overrides the default `__init__()` implementation."""

    var field: NotDefaultable

    def __init__(out self):
        self.field = NotDefaultable(7)


struct Parametrized[size: Int](Defaultable):
    """Derives a default `__init__()` for each parameterization.

    Parameters:
        size: The number of elements in the array field.
    """

    var array: Array[SimplePoint, Self.size]


def default_init[T: Defaultable]() -> T:
    return T()


def test_defaultable_simple() raises:
    """Tests the reflection-based default `__init__()` with a simple struct."""
    var p = SimplePoint()
    assert_equal(p.x, 0)
    assert_equal(p.y, 0)


def test_defaultable_nested() raises:
    """Tests the reflection-based default `__init__()` with nested structs."""
    var s = NestedStruct()
    assert_equal(s.point.x, 0)
    assert_equal(s.point.y, 0)
    assert_equal(s.name, "")


def test_defaultable_empty() raises:
    """Tests the reflection-based default `__init__()` with an empty struct."""
    var reflected = DefaultableEmpty()
    _ = reflected


def test_defaultable_override() raises:
    """Tests that an explicit `__init__()` bypasses the reflective default."""
    var o = Overridden()
    assert_equal(o.field.value, 7)


def test_defaultable_non_movable_field() raises:
    """Tests default initialization of a field that is not `Movable`."""
    var h = HasNonMovableField()
    assert_equal(h.field.value, 5)


def test_defaultable_array_field() raises:
    """Tests default initialization of a conditionally Defaultable field."""
    var h = HasArrayField()
    assert_equal(h.array[0].x, 0)
    assert_equal(h.array[1].y, 0)


def test_defaultable_parametrized() raises:
    """Tests the default `__init__()` with a parameterized struct."""
    var p = Parametrized[4]()
    assert_equal(p.array[3].x, 0)


def test_defaultable_generic() raises:
    """Tests default construction through a generic `Defaultable` bound."""
    var g = default_init[SimplePoint]()
    assert_equal(g.x, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
