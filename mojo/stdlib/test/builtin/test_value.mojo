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


from testing import assert_equal, assert_true
from utils import Variant

# ===-----------------------------------------------------------------------===#
# Triviality Struct
# ===-----------------------------------------------------------------------===#

alias EVENT_TRIVIAL = 0b1  # 1
alias EVENT_INIT = 0b10  # 2
alias EVENT_DEL = 0b100  # 4
alias EVENT_COPY = 0b1000  # 8
alias EVENT_MOVE = 0b10000  # 16


struct ConditionalTriviality[O: MutableOrigin, //, T: Movable & Copyable](
    Copyable, Movable
):
    var events: Pointer[List[Int], O]

    fn add_event(mut self, event: Int):
        self.events[].append(event)

    fn __init__(out self, ref [O]events: List[Int]):
        self.events = Pointer(to=events)
        self.add_event(EVENT_INIT)

    fn __del__(deinit self):
        @parameter
        if T.__del__is_trivial:
            self.add_event(EVENT_DEL | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_DEL)

    fn __copyinit__(out self, other: Self):
        self.events = other.events

        @parameter
        if T.__copyinit__is_trivial:
            self.add_event(EVENT_COPY | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_COPY)

    fn __moveinit__(out self, deinit other: Self):
        self.events = other.events

        @parameter
        if T.__moveinit__is_trivial:
            self.add_event(EVENT_MOVE | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_MOVE)


struct StructInheritTriviality[T: Movable & Copyable](Movable & Copyable):
    alias __moveinit__is_trivial = T.__moveinit__is_trivial
    alias __copyinit__is_trivial = T.__copyinit__is_trivial
    alias __del__is_trivial = T.__del__is_trivial


struct SpecificTrivialities[
    trivialities: Int = 0,
](Copyable & Movable):
    alias __moveinit__is_trivial = Bool(trivialities & EVENT_MOVE)
    alias __copyinit__is_trivial = Bool(trivialities & EVENT_COPY)
    alias __del__is_trivial = Bool(trivialities & EVENT_DEL)
    var events: UnsafePointer[List[Int]]

    fn __init__(out self, mut events: List[Int]):
        self.events = UnsafePointer(to=events)
        self.events[].append(EVENT_INIT)

    fn __copyinit__(out self, other: Self):
        self.events = other.events
        self.events[].append(EVENT_COPY)

    fn __moveinit__(out self, deinit other: Self):
        self.events = other.events
        self.events[].append(EVENT_MOVE)

    fn __del__(deinit self):
        self.events[].append(EVENT_DEL)


# ===-----------------------------------------------------------------------===#
# Individual tests
# ===-----------------------------------------------------------------------===#


def test_type_trivial[T: Movable & Copyable]():
    var events = List[Int]()
    var value = ConditionalTriviality[T](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
        ],
    )


def test_type_not_trivial[T: Movable & Copyable]():
    var events = List[Int]()
    var value = ConditionalTriviality[T](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events, [EVENT_INIT, EVENT_COPY, EVENT_DEL, EVENT_MOVE, EVENT_DEL]
    )


def test_type_inherit_triviality[T: Movable & Copyable]():
    var events = List[Int]()
    var value = ConditionalTriviality[StructInheritTriviality[T]](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
        ],
    )


def test_type_inherit_non_triviality[T: Movable & Copyable]():
    var events = List[Int]()
    var value = ConditionalTriviality[StructInheritTriviality[T]](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events, [EVENT_INIT, EVENT_COPY, EVENT_DEL, EVENT_MOVE, EVENT_DEL]
    )


# ===-----------------------------------------------------------------------===#
# Implementations tests
# ===-----------------------------------------------------------------------===#


def test_variant_specific_trivialities[trivialities: Int = 0]():
    alias type_specific_triviality = SpecificTrivialities[trivialities]
    alias variant_type = Variant[Int, Bool, type_specific_triviality]

    var expect_trivial_move = Bool(trivialities & EVENT_MOVE)
    var expect_trivial_copy = Bool(trivialities & EVENT_COPY)
    var expect_trivial_del = Bool(trivialities & EVENT_DEL)

    var result = List[Int]()
    var initial_specific_value = type_specific_triviality(result)
    assert_equal(result, [EVENT_INIT])

    var variant_value = variant_type(initial_specific_value^)
    assert_equal(result, [EVENT_INIT, EVENT_MOVE])

    # Variant now optimizes trivialities
    result.clear()
    var expected = List[Int]()

    var variant_copy = variant_value
    if not expect_trivial_copy:
        expected.append(EVENT_COPY)
    assert_equal(result, expected)

    var variant_moved = variant_value^
    if not expect_trivial_move:
        expected.append(EVENT_MOVE)
    assert_equal(result, expected)

    variant_moved^.__del__()
    variant_copy^.__del__()
    if not expect_trivial_del:
        expected.append(EVENT_DEL)
        expected.append(EVENT_DEL)
    assert_equal(result, expected)

    fn should_be_trivial(e: Int) -> Bool:
        return Bool(trivialities & e)

    if trivialities == 0:
        assert_true(EVENT_MOVE in result)
        assert_true(EVENT_COPY in result)
        assert_true(EVENT_DEL in result)
    else:
        for v in [EVENT_MOVE, EVENT_COPY, EVENT_DEL]:
            if should_be_trivial(v):
                assert_true(not v in result)
            else:
                assert_true(v in result)


def test_optional_specific_trivialities[trivialities: Int = 0]():
    alias type_specific_triviality = SpecificTrivialities[trivialities]
    alias optional_type = Optional[type_specific_triviality]

    var expect_trivial_move = Bool(trivialities & EVENT_MOVE)
    var expect_trivial_copy = Bool(trivialities & EVENT_COPY)
    var expect_trivial_del = Bool(trivialities & EVENT_DEL)

    var result = List[Int]()
    var initial_specific_value = type_specific_triviality(result)
    assert_equal(result, [EVENT_INIT])

    var optional_value = optional_type(initial_specific_value^)
    assert_equal(result, [EVENT_INIT, EVENT_MOVE])

    # Optional now optimizes trivialities
    result.clear()
    var expected = List[Int]()

    var optional_copy = optional_value
    if not expect_trivial_copy:
        expected.append(EVENT_COPY)
    assert_equal(result, expected)

    var optional_moved = optional_value^
    if not expect_trivial_move:
        expected.append(EVENT_MOVE)
    assert_equal(result, expected)

    optional_moved^.__del__()
    optional_copy^.__del__()
    if not expect_trivial_del:
        expected.append(EVENT_DEL)
        expected.append(EVENT_DEL)
    assert_equal(result, expected)

    fn should_be_trivial(e: Int) -> Bool:
        return Bool(trivialities & e)

    if trivialities == 0:
        assert_true(EVENT_MOVE in result)
        assert_true(EVENT_COPY in result)
        assert_true(EVENT_DEL in result)
    else:
        for v in [EVENT_MOVE, EVENT_COPY, EVENT_DEL]:
            if should_be_trivial(v):
                assert_true(not v in result)
            else:
                assert_true(v in result)


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main():
    test_type_trivial[Int]()
    test_type_not_trivial[String]()
    test_type_inherit_triviality[Float64]()
    test_type_inherit_non_triviality[String]()

    # test_type_inherit_triviality[InlineArray[InlineArray[Int, 4], 4]]()
    # test_type_inherit_non_triviality[InlineArray[InlineArray[String, 4], 4]]()

    # Integrate into Variant
    test_type_trivial[Variant[Int, Bool]]()
    test_type_not_trivial[Variant[String, Bool]]()

    # Integrate into Optional
    test_type_trivial[Optional[Int]]()
    test_type_not_trivial[Optional[String]]()

    # test_type_inherit_triviality[InlineArray[Optional[Int], 4]]()
    # test_type_inherit_non_triviality[InlineArray[Optional[String], 4]]()

    test_variant_specific_trivialities()
    test_variant_specific_trivialities[EVENT_MOVE]()
    test_variant_specific_trivialities[EVENT_COPY]()
    test_variant_specific_trivialities[EVENT_DEL]()
    test_variant_specific_trivialities[EVENT_DEL | EVENT_COPY | EVENT_MOVE]()

    test_optional_specific_trivialities()
    test_optional_specific_trivialities[EVENT_MOVE]()
    test_optional_specific_trivialities[EVENT_COPY]()
    test_optional_specific_trivialities[EVENT_DEL]()
    test_optional_specific_trivialities[EVENT_DEL | EVENT_COPY | EVENT_MOVE]()
