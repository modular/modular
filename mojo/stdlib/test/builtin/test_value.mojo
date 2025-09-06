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


from testing import assert_false, assert_true, assert_equal

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
# Main
# ===-----------------------------------------------------------------------===#


def main():
    test_type_trivial[Int]()
    test_type_not_trivial[String]()
    test_type_inherit_triviality[Float64]()
    test_type_inherit_non_triviality[String]()
    # test_type_inherit_triviality[InlineArray[InlineArray[Int, 4], 4]]()
    # test_type_inherit_non_triviality[InlineArray[InlineArray[String, 4], 4]]()
