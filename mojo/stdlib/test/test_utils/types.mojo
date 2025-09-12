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
"""Types for testing object lifecycle events.

* `MoveCounter`
* `CopyCounter`
* `MoveCopyCounter`
* `DelCounter`
* `CopyCountedStruct`
* `MoveOnly`
* `ExplicitCopyOnly`
* `ImplicitCopyOnly`
* `ObservableMoveOnly`
* `ObservableDel`
* `DelRecorder`
* `AbortOnDel`
* `AbortOnCopy`
"""

from os import abort


# ===----------------------------------------------------------------------=== #
# MoveOnly
# ===----------------------------------------------------------------------=== #


struct MoveOnly[T: Movable](Movable):
    """Utility for testing MoveOnly types.

    Parameters:
        T: Can be any type satisfying the Movable trait.
    """

    var data: T
    """Test data payload."""

    @implicit
    fn __init__(out self, var i: T):
        """Construct a MoveOnly providing the payload data.

        Args:
            i: The test data payload.
        """
        self.data = i^

    fn __moveinit__(out self, deinit other: Self):
        """Move construct a MoveOnly from an existing variable.

        Args:
            other: The other instance that we copying the payload from.
        """
        self.data = other.data^


# ===----------------------------------------------------------------------=== #
# ObservableMoveOnly
# ===----------------------------------------------------------------------=== #


struct ObservableMoveOnly[actions_origin: ImmutableOrigin](Movable):
    alias _U = UnsafePointer[List[String], mut=False, origin=actions_origin]
    var actions: Self._U
    var value: Int

    fn __init__(out self, value: Int, actions: Self._U):
        self.actions = actions
        self.value = value
        self.actions.origin_cast[True]()[0].append("__init__")

    fn __moveinit__(out self, deinit existing: Self):
        self.actions = existing.actions
        self.value = existing.value
        self.actions.origin_cast[True]()[0].append("__moveinit__")

    fn __del__(deinit self):
        self.actions.origin_cast[True]()[0].append("__del__")


# ===----------------------------------------------------------------------=== #
# ExplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ExplicitCopyOnly(Copyable):
    var value: Int
    var copy_count: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value
        self.copy_count = 0

    fn __copyinit__(out self, other: Self):
        self = Self(other.value)
        self.copy_count = other.copy_count + 1


# ===----------------------------------------------------------------------=== #
# ImplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ImplicitCopyOnly(ImplicitlyCopyable):
    var value: Int
    var copy_count: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value
        self.copy_count = 0

    fn __copyinit__(out self, other: Self):
        self.value = other.value
        self.copy_count = other.copy_count + 1


# ===----------------------------------------------------------------------=== #
# CopyCounter
# ===----------------------------------------------------------------------=== #


struct CopyCounter(ImplicitlyCopyable, Movable, Writable):
    """Counts the number of copies performed on a value."""

    var copy_count: Int

    fn __init__(out self):
        self.copy_count = 0

    fn __moveinit__(out self, deinit existing: Self):
        self.copy_count = existing.copy_count

    fn __copyinit__(out self, existing: Self):
        self.copy_count = existing.copy_count + 1

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("CopyCounter(", self.copy_count, ")")


# ===----------------------------------------------------------------------=== #
# MoveCounter
# ===----------------------------------------------------------------------=== #


struct MoveCounter[T: Copyable & Movable](ImplicitlyCopyable, Movable):
    """Counts the number of moves performed on a value."""

    var value: T
    var move_count: Int

    @implicit
    fn __init__(out self, var value: T):
        """Construct a new instance of this type. This initial move is not counted.
        """
        self.value = value^
        self.move_count = 0

    fn __moveinit__(out self, deinit existing: Self):
        self.value = existing.value^
        self.move_count = existing.move_count + 1

    # TODO: This type should not be Copyable, but has to be to satisfy
    #       Copyable & Movable at the moment.
    fn __copyinit__(out self, existing: Self):
        self.value = existing.value.copy()
        self.move_count = existing.move_count


# ===----------------------------------------------------------------------=== #
# MoveCopyCounter
# ===----------------------------------------------------------------------=== #


struct MoveCopyCounter(ImplicitlyCopyable, Movable):
    var copied: Int
    var moved: Int

    fn __init__(out self):
        self.copied = 0
        self.moved = 0

    fn __copyinit__(out self, other: Self):
        self.copied = other.copied + 1
        self.moved = other.moved

    fn __moveinit__(out self, deinit other: Self):
        self.copied = other.copied
        self.moved = other.moved + 1


# ===----------------------------------------------------------------------=== #
# DelRecorder
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelRecorder[recorder_origin: ImmutableOrigin](
    ImplicitlyCopyable, Movable
):
    var value: Int
    var destructor_recorder: UnsafePointer[
        List[Int], mut=False, origin=recorder_origin
    ]

    fn __del__(deinit self):
        self.destructor_recorder.origin_cast[True]()[].append(self.value)

    fn copy(self) -> Self:
        return Self(self.value, self.destructor_recorder)


# ===----------------------------------------------------------------------=== #
# ObservableDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ObservableDel[origin: MutableOrigin = MutableAnyOrigin](
    ImplicitlyCopyable, Movable
):
    var target: UnsafePointer[Bool, origin=origin]

    fn __del__(deinit self):
        self.target.init_pointee_move(True)


# ===----------------------------------------------------------------------=== #
# DelCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelCounter[counter_origin: ImmutableOrigin](
    ImplicitlyCopyable, Movable, Writable
):
    var counter: UnsafePointer[Int, mut=False, origin=counter_origin]

    fn __del__(deinit self):
        self.counter.origin_cast[True]()[] += 1

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("DelCounter(", self.counter[], ")")


# ===----------------------------------------------------------------------=== #
# AbortOnDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnDel(ImplicitlyCopyable, Movable):
    var value: Int

    fn __del__(deinit self):
        abort("We should never call the destructor of AbortOnDel")


# ===----------------------------------------------------------------------=== #
# CopyCountedStruct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CopyCountedStruct(ImplicitlyCopyable, Movable):
    var counter: CopyCounter
    var value: String

    @implicit
    fn __init__(out self, value: String):
        self.counter = CopyCounter()
        self.value = value


# ===----------------------------------------------------------------------=== #
# AbortOnCopy
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnCopy(ImplicitlyCopyable):
    fn __copyinit__(out self, other: Self):
        abort("We should never implicitly copy AbortOnCopy")
