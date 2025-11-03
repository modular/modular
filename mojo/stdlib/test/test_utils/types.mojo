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
* `TriviallyCopyableMoveCounter`
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


# ===----------------------------------------------------------------------=== #
# ObservableMoveOnly
# ===----------------------------------------------------------------------=== #


struct ObservableMoveOnly[actions_origin: ImmutOrigin](Movable):
    """Type for observing move and destruction operations during testing.

    Parameters:
        actions_origin: Origin of the actions list for tracking operations.
    """

    alias _U = UnsafePointer[List[String], mut=False, origin=actions_origin]
    var actions: Self._U
    """Pointer to list tracking lifecycle operations."""
    var value: Int
    """Test value payload."""

    fn __init__(out self, value: Int, actions: Self._U):
        """Constructs a new instance and records the initialization.

        Args:
            value: The value to store.
            actions: Pointer to list for recording operations.
        """
        self.actions = actions
        self.value = value
        self.actions.unsafe_mut_cast[True]()[0].append("__init__")

    fn __moveinit__(out self, deinit existing: Self):
        """Moves from an existing instance and records the operation.

        Args:
            existing: The instance being moved from.
        """
        self.actions = existing.actions
        self.value = existing.value
        self.actions.unsafe_mut_cast[True]()[0].append("__moveinit__")

    fn __del__(deinit self):
        """Destroys the instance and records the operation."""
        self.actions.unsafe_mut_cast[True]()[0].append("__del__")


# ===----------------------------------------------------------------------=== #
# ExplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ExplicitCopyOnly(Copyable):
    """Type with explicit copy semantics for testing."""

    var value: Int
    """Integer value payload."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    @implicit
    fn __init__(out self, value: Int):
        """Constructs a new instance.

        Args:
            value: The integer value to store.
        """
        self.value = value
        self.copy_count = 0

    fn __copyinit__(out self, other: Self):
        """Copies from another instance and increments the count.

        Args:
            other: The instance being copied from.
        """
        self = Self(other.value)
        self.copy_count = other.copy_count + 1


# ===----------------------------------------------------------------------=== #
# ImplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ImplicitCopyOnly(ImplicitlyCopyable):
    """Type with implicit copy semantics for testing."""

    var value: Int
    """Integer value payload."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    @implicit
    fn __init__(out self, value: Int):
        """Constructs a new instance.

        Args:
            value: The integer value to store.
        """
        self.value = value
        self.copy_count = 0

    fn __copyinit__(out self, other: Self):
        """Copies from another instance and increments the count.

        Args:
            other: The instance being copied from.
        """
        self.value = other.value
        self.copy_count = other.copy_count + 1


# ===----------------------------------------------------------------------=== #
# CopyCounter
# ===----------------------------------------------------------------------=== #


struct CopyCounter[
    T: ImplicitlyCopyable & Movable & Writable & Defaultable = NoneType
](ImplicitlyCopyable, Movable, Writable):
    """Counts the number of copies performed on a value.

    Parameters:
        T: The type of value to wrap and count copies for.
    """

    var value: T
    """The wrapped value."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    fn __init__(out self):
        """Constructs a new instance with default value."""
        self = Self(T())

    fn __init__(out self, s: T):
        """Constructs a new instance with the given value.

        Args:
            s: The value to wrap.
        """
        self.value = s
        self.copy_count = 0

    fn __copyinit__(out self, existing: Self):
        """Copies from another instance and increments the count.

        Args:
            existing: The instance being copied from.
        """
        self.value = existing.value
        self.copy_count = existing.copy_count + 1

    fn write_to(self, mut writer: Some[Writer]):
        """Writes a string representation to the writer.

        Args:
            writer: The writer to output to.
        """
        writer.write("CopyCounter(", self.value, " ", self.copy_count, ")")


# ===----------------------------------------------------------------------=== #
# MoveCounter
# ===----------------------------------------------------------------------=== #


# TODO: This type should not be Copyable, but has to be to satisfy
#       Copyable & Movable at the moment.
struct MoveCounter[T: Copyable & Movable](Copyable, Movable):
    """Counts the number of moves performed on a value.

    Parameters:
        T: The type of value to wrap and count moves for.
    """

    var value: T
    """The wrapped value."""
    var move_count: Int
    """Number of times this instance has been moved."""

    @implicit
    fn __init__(out self, var value: T):
        """Constructs a new instance of this type. This initial move is not counted.

        Args:
            value: The value to wrap.
        """
        self.value = value^
        self.move_count = 0

    fn __moveinit__(out self, deinit existing: Self):
        """Moves from an existing instance and increments the count.

        Args:
            existing: The instance being moved from.
        """
        self.value = existing.value^
        self.move_count = existing.move_count + 1


# ===----------------------------------------------------------------------=== #
# MoveCopyCounter
# ===----------------------------------------------------------------------=== #


struct MoveCopyCounter(ImplicitlyCopyable, Movable):
    """Counts both copy and move operations for testing."""

    var copied: Int
    """Number of times this instance has been copied."""
    var moved: Int
    """Number of times this instance has been moved."""

    fn __init__(out self):
        """Constructs a new instance with zero counts."""
        self.copied = 0
        self.moved = 0

    fn __copyinit__(out self, other: Self):
        """Copies from another instance and increments the copy count.

        Args:
            other: The instance being copied from.
        """
        self.copied = other.copied + 1
        self.moved = other.moved

    fn __moveinit__(out self, deinit other: Self):
        """Moves from an existing instance and increments the move count.

        Args:
            other: The instance being moved from.
        """
        self.copied = other.copied
        self.moved = other.moved + 1


# ===----------------------------------------------------------------------=== #
# TriviallyCopyableMoveCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TriviallyCopyableMoveCounter(Copyable, Movable):
    """Type used for testing that collections still perform moves and not copies
    when a type has a custom __moveinit__() but is also trivially copyable.

    Types with this property are rare in practice, but its still important to
    get the modeling right. If a type author wants their type to be moved
    with a memcpy, they should mark it as such."""

    var move_count: Int
    """Number of times this instance has been moved."""

    # Copying this type is trivial, it doesn't care to track copies.
    alias __copyinit__is_trivial = True

    fn __moveinit__(out self, deinit existing: Self):
        """Moves from an existing instance and increments the count.

        Args:
            existing: The instance being moved from.
        """
        self.move_count = existing.move_count + 1


# ===----------------------------------------------------------------------=== #
# DelRecorder
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelRecorder[recorder_origin: ImmutOrigin](ImplicitlyCopyable, Movable):
    """Records destructor calls for testing.

    Parameters:
        recorder_origin: Origin of the recorder list.
    """

    var value: Int
    """Value to record when destroyed."""
    var destructor_recorder: UnsafePointer[
        List[Int], mut=False, origin=recorder_origin
    ]
    """Pointer to list for recording destructor calls."""

    fn __del__(deinit self):
        """Records this instance's value when destroyed."""
        self.destructor_recorder.unsafe_mut_cast[True]()[].append(self.value)

    fn copy(self) -> Self:
        """Creates a copy of this instance.

        Returns:
            A new instance with the same value and recorder.
        """
        return Self(self.value, self.destructor_recorder)


# ===----------------------------------------------------------------------=== #
# ObservableDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ObservableDel[origin: MutOrigin = MutAnyOrigin](
    ImplicitlyCopyable, Movable
):
    """Sets a boolean flag when destroyed.

    Parameters:
        origin: Origin of the target pointer.
    """

    var target: UnsafePointer[Bool, origin=origin]
    """Pointer to boolean flag set on destruction."""

    fn __del__(deinit self):
        """Sets the target flag to True when destroyed."""
        self.target.init_pointee_move(True)


# ===----------------------------------------------------------------------=== #
# DelCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelCounter[counter_origin: ImmutOrigin, *, trivial_del: Bool = False](
    ImplicitlyCopyable, Movable, Writable
):
    """Counts the number of times instances are destroyed.

    Parameters:
        counter_origin: Origin of the counter pointer.
        trivial_del: Whether the destructor is trivial.
    """

    alias __del__is_trivial = trivial_del

    var counter: UnsafePointer[Int, mut=False, origin=counter_origin]
    """Pointer to counter incremented on destruction."""

    fn __del__(deinit self):
        """Increments the counter when destroyed."""
        self.counter.unsafe_mut_cast[True]()[] += 1

    fn write_to(self, mut writer: Some[Writer]):
        """Writes a string representation to the writer.

        Args:
            writer: The writer to output to.
        """
        writer.write("DelCounter(", self.counter[], ")")


# ===----------------------------------------------------------------------=== #
# AbortOnDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnDel(ImplicitlyCopyable, Movable):
    """Type that aborts if its destructor is called.

    Used to test that destructors are not called in certain scenarios.
    """

    var value: Int
    """Test value payload."""

    fn __del__(deinit self):
        """Aborts the program if called."""
        abort("We should never call the destructor of AbortOnDel")


# ===----------------------------------------------------------------------=== #
# CopyCountedStruct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CopyCountedStruct(ImplicitlyCopyable, Movable):
    """Struct that tracks the number of times it has been copied."""

    var counter: CopyCounter
    """Counter tracking copy operations."""
    var value: String
    """String value payload."""

    @implicit
    fn __init__(out self, value: String):
        """Constructs a new instance with the given value.

        Args:
            value: The string value to store.
        """
        self.counter = CopyCounter()
        self.value = value


# ===----------------------------------------------------------------------=== #
# AbortOnCopy
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnCopy(ImplicitlyCopyable):
    """Type that aborts if copied.

    Used to test that implicit copies do not occur in certain scenarios.
    """

    fn __copyinit__(out self, other: Self):
        """Aborts the program if called.

        Args:
            other: The instance being copied from.
        """
        abort("We should never implicitly copy AbortOnCopy")
