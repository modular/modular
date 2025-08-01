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

    fn __moveinit__(out self, owned other: Self):
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
        self.actions.origin_cast[mut=True]()[0].append("__init__")

    fn __moveinit__(out self, owned existing: Self):
        self.actions = existing.actions
        self.value = existing.value
        self.actions.origin_cast[mut=True]()[0].append("__moveinit__")

    fn __del__(owned self):
        self.actions.origin_cast[mut=True]()[0].append("__del__")


# ===----------------------------------------------------------------------=== #
# ExplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ExplicitCopyOnly(ExplicitlyCopyable):
    var value: Int
    var copy_count: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value
        self.copy_count = 0

    fn copy(self, out copy: Self):
        copy = Self(self.value)
        copy.copy_count = self.copy_count + 1


# ===----------------------------------------------------------------------=== #
# ImplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ImplicitCopyOnly(Copyable):
    var value: Int
    var copy_count: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value
        self.copy_count = 0

    fn __copyinit__(out self, *, other: Self):
        self.value = other.value
        self.copy_count = other.copy_count + 1


# ===----------------------------------------------------------------------=== #
# CopyCounter
# ===----------------------------------------------------------------------=== #


struct CopyCounter(Copyable, ExplicitlyCopyable, Movable, Writable):
    """Counts the number of copies performed on a value."""

    var copy_count: Int

    fn __init__(out self):
        self.copy_count = 0

    fn __init__(out self, *, other: Self):
        self.copy_count = other.copy_count + 1

    fn __moveinit__(out self, owned existing: Self):
        self.copy_count = existing.copy_count

    fn __copyinit__(out self, existing: Self):
        self.copy_count = existing.copy_count + 1

    fn copy(self) -> Self:
        return self

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("CopyCounter(", self.copy_count, ")")


# ===----------------------------------------------------------------------=== #
# MoveCounter
# ===----------------------------------------------------------------------=== #


struct MoveCounter[T: ExplicitlyCopyable & Movable](
    Copyable, ExplicitlyCopyable, Movable
):
    """Counts the number of moves performed on a value."""

    var value: T
    var move_count: Int

    @implicit
    fn __init__(out self, var value: T):
        """Construct a new instance of this type. This initial move is not counted.
        """
        self.value = value^
        self.move_count = 0

    # TODO: This type should not be ExplicitlyCopyable, but has to be to satisfy
    #       CollectionElementNew at the moment.
    fn __init__(out self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.value = other.value.copy()
        self.move_count = other.move_count

    fn __moveinit__(out self, owned existing: Self):
        self.value = existing.value^
        self.move_count = existing.move_count + 1

    # TODO: This type should not be Copyable, but has to be to satisfy
    #       Copyable & Movable at the moment.
    fn __copyinit__(out self, existing: Self):
        self.value = existing.value.copy()
        self.move_count = existing.move_count

    fn copy(self, out existing: Self):
        existing = Self(self.value.copy())
        existing.move_count = self.move_count


# ===----------------------------------------------------------------------=== #
# MoveCopyCounter
# ===----------------------------------------------------------------------=== #


struct MoveCopyCounter(Copyable, Movable):
    var copied: Int
    var moved: Int

    fn __init__(out self):
        self.copied = 0
        self.moved = 0

    fn __copyinit__(out self, other: Self):
        self.copied = other.copied + 1
        self.moved = other.moved

    fn copy(self) -> Self:
        return self

    fn __moveinit__(out self, owned other: Self):
        self.copied = other.copied
        self.moved = other.moved + 1


# ===----------------------------------------------------------------------=== #
# DelRecorder
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelRecorder[recorder_origin: ImmutableOrigin](
    Copyable, ExplicitlyCopyable, Movable
):
    var value: Int
    var destructor_recorder: UnsafePointer[
        List[Int], mut=False, origin=recorder_origin
    ]

    fn __del__(owned self):
        self.destructor_recorder.origin_cast[mut=True]()[].append(self.value)

    fn copy(self) -> Self:
        return Self(self.value, self.destructor_recorder)


# ===----------------------------------------------------------------------=== #
# ObservableDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ObservableDel[origin: MutableOrigin = MutableAnyOrigin](
    Copyable & Movable
):
    var target: UnsafePointer[Bool, origin=origin]

    fn __init__(out self, *, other: Self):
        self = other

    fn __del__(owned self):
        self.target.init_pointee_move(True)


# ===----------------------------------------------------------------------=== #
# DelCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelCounter[counter_origin: ImmutableOrigin](Copyable, Movable, Writable):
    var counter: UnsafePointer[Int, mut=False, origin=counter_origin]

    fn __del__(owned self):
        self.counter.origin_cast[mut=True]()[] += 1

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("DelCounter(", self.counter[], ")")


# ===----------------------------------------------------------------------=== #
# AbortOnDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnDel(Copyable, Movable):
    var value: Int

    fn __del__(owned self):
        abort("We should never call the destructor of AbortOnDel")


# ===----------------------------------------------------------------------=== #
# CopyCountedStruct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CopyCountedStruct(Copyable, Movable):
    var counter: CopyCounter
    var value: String

    fn __init__(out self, *, other: Self):
        self.counter = other.counter.copy()
        self.value = other.value.copy()

    @implicit
    fn __init__(out self, value: String):
        self.counter = CopyCounter()
        self.value = value


# ===----------------------------------------------------------------------=== #
# AbortOnCopy
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnCopy(Copyable, ExplicitlyCopyable):
    fn __copyinit__(out self, other: Self):
        abort("We should never implicitly copy AbortOnCopy")

    fn copy(self) -> Self:
        abort("We should never explicitly copy AbortOnCopy")
        return self
