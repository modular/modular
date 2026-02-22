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


from testing import assert_equal, assert_false, assert_true, TestSuite

# ===-----------------------------------------------------------------------===#
# Triviality Struct
# ===-----------------------------------------------------------------------===#

comptime EVENT_TRIVIAL = 0b1  # 1
comptime EVENT_INIT = 0b10  # 2
comptime EVENT_DEL = 0b100  # 4
comptime EVENT_COPY = 0b1000  # 8
comptime EVENT_MOVE = 0b10000  # 16


struct ConditionalTriviality[
    O: MutOrigin,
    //,
    T: Copyable & ImplicitlyDestructible,
](Copyable):
    var events: Pointer[List[Int], Self.O]

    fn add_event(mut self, event: Int):
        self.events[].append(event)

    fn __init__(out self, ref[Self.O] events: List[Int]):
        self.events = Pointer(to=events)
        self.add_event(EVENT_INIT)

    fn __del__(deinit self):
        comptime if Self.T.__del__is_trivial:
            self.add_event(EVENT_DEL | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_DEL)

    fn __init__(out self, *, copy: Self):
        self.events = copy.events

        comptime if Self.T.__copyinit__is_trivial:
            self.add_event(EVENT_COPY | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_COPY)

    fn __init__(out self, *, deinit take: Self):
        self.events = take.events

        comptime if Self.T.__moveinit__is_trivial:
            self.add_event(EVENT_MOVE | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_MOVE)


struct StructInheritTriviality[T: Copyable & ImplicitlyDestructible](Copyable):
    comptime __moveinit__is_trivial = Self.T.__moveinit__is_trivial
    comptime __copyinit__is_trivial = Self.T.__copyinit__is_trivial
    comptime __del__is_trivial = Self.T.__del__is_trivial


# ===-----------------------------------------------------------------------===#
# Individual tests
# ===-----------------------------------------------------------------------===#


def _test_type_trivial[T: Copyable & ImplicitlyDestructible]():
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


def test_type_trivial():
    _test_type_trivial[Int]()


def _test_type_not_trivial[T: Copyable & ImplicitlyDestructible]():
    var events = List[Int]()
    var value = ConditionalTriviality[T](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY,
            EVENT_DEL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL,
        ],
    )


def test_type_not_trivial():
    _test_type_not_trivial[String]()


def _test_type_inherit_triviality[T: Copyable & ImplicitlyDestructible]():
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


def test_type_inherit_triviality():
    _test_type_inherit_triviality[Float64]()
    # _test_type_inherit_triviality[InlineArray[InlineArray[Int, 4], 4]]()


def _test_type_inherit_non_triviality[T: Copyable & ImplicitlyDestructible]():
    var events = List[Int]()
    var value = ConditionalTriviality[StructInheritTriviality[T]](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY,
            EVENT_DEL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL,
        ],
    )


def test_type_inherit_non_triviality():
    _test_type_inherit_non_triviality[String]()
    # _test_type_inherit_non_triviality[InlineArray[InlineArray[String, 4], 4]]()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
