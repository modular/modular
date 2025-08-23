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
from memory import memcpy

# ===-----------------------------------------------------------------------===#
# Triviality Struct
# ===-----------------------------------------------------------------------===#

alias event_trivial = 1
alias event_init = 2
alias event_del = 4
alias event_copy = 8
alias event_move = 16


struct ConditionalTriviality[T: Movable & Copyable](Copyable, Movable):
    var events: UnsafePointer[List[Int]]

    fn add_event(self, event: Int):
        self.events[].append(event)

    fn __init__(out self, mut events: List[Int]):
        self.events = UnsafePointer(to=events)
        self.add_event(event_init)

    fn __del__(deinit self):
        @parameter
        if T.__del__is_trivial:
            self.add_event(event_del | event_trivial)
        else:
            self.add_event(event_del)

    fn __copyinit__(out self, other: Self):
        self.events = other.events

        @parameter
        if T.__copyinit__is_trivial:
            self.add_event(event_copy | event_trivial)
        else:
            self.add_event(event_copy)

    fn __moveinit__(out self, deinit other: Self):
        self.events = other.events

        @parameter
        if T.__moveinit__is_trivial:
            self.add_event(event_move | event_trivial)
        else:
            self.add_event(event_move)


# ===-----------------------------------------------------------------------===#
# Individual tests
# ===-----------------------------------------------------------------------===#


def test_type_trivial():
    var events = List[Int]()
    var value = ConditionalTriviality[Int](events)
    var value_copy = value
    # ^ optimized copy->move
    # keep it:
    value^.__del__()
    var value_move = value_copy^
    assert_equal(
        events,
        [
            event_init,
            event_copy | event_trivial,
            event_del | event_trivial,
            event_move | event_trivial,
            event_del | event_trivial,
        ],
    )


def test_type_not_trivial():
    var events = List[Int]()
    var value = ConditionalTriviality[String](events)
    var value_copy = value
    # ^ optimized copy->move
    # keep it:
    value^.__del__()
    var value_move = value_copy^
    assert_equal(
        events, [event_init, event_copy, event_del, event_move, event_del]
    )


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main():
    test_type_trivial()
    test_type_not_trivial()
