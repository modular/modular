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


struct MoveOnly[T: Movable & ImplicitlyDestructible](Movable):
    """Utility for testing MoveOnly types.

    Parameters:
        T: Can be any type satisfying the Movable trait.
    """

    var data: Self.T
    """Test data payload."""

    @implicit
    fn __init__(out self, var i: Self.T):
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

    comptime _U = UnsafePointer[List[String], Self.actions_origin]
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

    fn __moveinit__(out self, deinit take: Self):
        """Moves from an existing instance and records the operation.

        Args:
            take: The instance being moved from.
        """
        self.actions = take.actions
        self.value = take.value
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

    fn __copyinit__(out self, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self = Self(copy.value)
        self.copy_count = copy.copy_count + 1


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

    fn __copyinit__(out self, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self.value = copy.value
        self.copy_count = copy.copy_count + 1


# ===----------------------------------------------------------------------=== #
# CopyCounter
# ===----------------------------------------------------------------------=== #


struct CopyCounter[
    T: ImplicitlyCopyable & Writable & Defaultable = NoneType,
    *,
    trivial_copy: Bool = False,
](ImplicitlyCopyable, Writable):
    """Counts the number of copies performed on a value.

    Parameters:
        T: The type of value to wrap and count copies for.
        trivial_copy: Weather the copy constructor should be considered trivial.
    """

    comptime __copyinit__is_trivial = Self.trivial_copy

    var value: Self.T
    """The wrapped value."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    fn __init__(out self):
        """Constructs a new instance with default value."""
        self = Self(Self.T())

    fn __init__(out self, s: Self.T):
        """Constructs a new instance with the given value.

        Args:
            s: The value to wrap.
        """
        comptime assert (
            Self.T.__copyinit__is_trivial or not Self.trivial_copy
        ), (
            "You cannot override CopyCounter's trivial copy construct when T"
            " does not have one"
        )
        self.value = s
        self.copy_count = 0

    fn __copyinit__(out self, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self.value = copy.value
        self.copy_count = copy.copy_count + 1

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
#       Copyable at the moment.
struct MoveCounter[
    T: Copyable & ImplicitlyDestructible, *, trivial_move: Bool = False
](Copyable):
    """Counts the number of moves performed on a value.

    Parameters:
        T: The type of value to wrap and count moves for.
        trivial_move: Weather the move constructor should be treated as trivial.
    """

    comptime __moveinit__is_trivial = Self.trivial_move

    var value: Self.T
    """The wrapped value."""
    var move_count: Int
    """Number of times this instance has been moved."""

    @implicit
    fn __init__(out self, var value: Self.T):
        """Constructs a new instance of this type. This initial move is not counted.

        Args:
            value: The value to wrap.
        """
        comptime assert (
            Self.T.__moveinit__is_trivial or not Self.trivial_move
        ), (
            "You cannot override MoveCounter's trivial move construct when T"
            " does not have one"
        )
        self.value = value^
        self.move_count = 0

    fn __moveinit__(out self, deinit take: Self):
        """Moves from an existing instance and increments the count.

        Args:
            take: The instance being moved from.
        """
        self.value = take.value^
        self.move_count = take.move_count + 1


# ===----------------------------------------------------------------------=== #
# MoveCopyCounter
# ===----------------------------------------------------------------------=== #


struct MoveCopyCounter(ImplicitlyCopyable):
    """Counts both copy and move operations for testing."""

    var copied: Int
    """Number of times this instance has been copied."""
    var moved: Int
    """Number of times this instance has been moved."""

    fn __init__(out self):
        """Constructs a new instance with zero counts."""
        self.copied = 0
        self.moved = 0

    fn __copyinit__(out self, copy: Self):
        """Copies from another instance and increments the copy count.

        Args:
            copy: The instance being copied from.
        """
        self.copied = copy.copied + 1
        self.moved = copy.moved

    fn __moveinit__(out self, deinit take: Self):
        """Moves from an existing instance and increments the move count.

        Args:
            take: The instance being moved from.
        """
        self.copied = take.copied
        self.moved = take.moved + 1


# ===----------------------------------------------------------------------=== #
# TriviallyCopyableMoveCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TriviallyCopyableMoveCounter(Copyable):
    """Type used for testing that collections still perform moves and not copies
    when a type has a custom __moveinit__() but is also trivially copyable.

    Types with this property are rare in practice, but its still important to
    get the modeling right. If a type author wants their type to be moved
    with a memcpy, they should mark it as such."""

    var move_count: Int
    """Number of times this instance has been moved."""

    # Copying this type is trivial, it doesn't care to track copies.
    comptime __copyinit__is_trivial = True

    fn __moveinit__(out self, deinit take: Self):
        """Moves from an existing instance and increments the count.

        Args:
            take: The instance being moved from.
        """
        self.move_count = take.move_count + 1


# ===----------------------------------------------------------------------=== #
# DelRecorder
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelRecorder[recorder_origin: ImmutOrigin](ImplicitlyCopyable):
    """Records destructor calls for testing.

    Parameters:
        recorder_origin: Origin of the recorder list.
    """

    var value: Int
    """Value to record when destroyed."""
    var destructor_recorder: UnsafePointer[List[Int], Self.recorder_origin]
    """Pointer to list for recording destructor calls."""

    fn __del__(deinit self):
        """Records this instance's value when destroyed."""
        self.destructor_recorder.unsafe_mut_cast[True]()[].append(self.value)


# ===----------------------------------------------------------------------=== #
# ObservableDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ObservableDel[origin: MutOrigin = MutAnyOrigin](ImplicitlyCopyable):
    """Sets a boolean flag when destroyed.

    Parameters:
        origin: Origin of the target pointer.
    """

    var target: UnsafePointer[Bool, Self.origin]
    """Pointer to boolean flag set on destruction."""

    fn __del__(deinit self):
        """Sets the target flag to True when destroyed."""
        self.target.init_pointee_move(True)


# ===----------------------------------------------------------------------=== #
# ExplicitDelOnly
# ===----------------------------------------------------------------------=== #


@explicit_destroy
@fieldwise_init
struct ExplicitDelOnly(Movable):
    """Utility for testing container support for linear types."""

    var data: Int
    """Test data payload."""

    fn destroy(deinit self):
        """Explicitly destroy this linear type."""

        pass


# ===----------------------------------------------------------------------=== #
# DelCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelCounter[counter_origin: ImmutOrigin, *, trivial_del: Bool = False](
    ImplicitlyCopyable, Writable
):
    """Counts the number of times instances are destroyed.

    Parameters:
        counter_origin: Origin of the counter pointer.
        trivial_del: Whether the destructor is trivial.
    """

    comptime __del__is_trivial = Self.trivial_del

    var counter: UnsafePointer[Int, Self.counter_origin]
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
struct AbortOnDel(ImplicitlyCopyable):
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
struct CopyCountedStruct(ImplicitlyCopyable):
    """Struct that tracks the number of times it has been copied."""

    var counter: CopyCounter[]
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

    fn __copyinit__(out self, copy: Self):
        """Aborts the program if called.

        Args:
            copy: The instance being copied from.
        """
        abort("We should never implicitly copy AbortOnCopy")


# ===----------------------------------------------------------------------=== #
# Observable
# ===----------------------------------------------------------------------=== #


struct Observable[
    *,
    CopyOrigin: MutOrigin = MutExternalOrigin,
    MoveOrigin: MutOrigin = MutExternalOrigin,
    DelOrigin: MutOrigin = MutExternalOrigin,
](Copyable):
    """A type that tracks the number of times it has been copied, moved, and destroyed.

    Parameters:
        CopyOrigin: Origin of the copies pointer.
        MoveOrigin: Origin of the moves pointer.
        DelOrigin: Origin of the dels pointer.
    """

    var _copies: Optional[Pointer[Int, Self.CopyOrigin]]
    var _moves: Optional[Pointer[Int, Self.MoveOrigin]]
    var _dels: Optional[Pointer[Int, Self.DelOrigin]]

    fn __init__(
        out self,
        *,
        var copies: Optional[Pointer[Int, Self.CopyOrigin]] = None,
        var moves: Optional[Pointer[Int, Self.MoveOrigin]] = None,
        var dels: Optional[Pointer[Int, Self.DelOrigin]] = None,
    ):
        """Constructs a new Observable with the given pointers.

        Args:
            copies: Optional pointer to an Int to count copies.
            moves: Optional pointer to an Int to count moves.
            dels: Optional pointer to an Int to count dels.
        """
        self._copies = copies^
        self._moves = moves^
        self._dels = dels^

    fn __copyinit__(out self, copy: Self):
        """Copy initialize the Observable and increment the copy count.

        Args:
            copy: The instance being copied from.
        """
        self._copies = copy._copies.copy()
        self._moves = copy._moves.copy()
        self._dels = copy._dels.copy()
        if self._copies:
            self._copies.value()[] += 1

    fn __moveinit__(out self, deinit take: Self):
        """Move initialize the Observable and increment the move count.

        Args:
            take: The instance being moved from.
        """
        self._copies = take._copies^
        self._moves = take._moves^
        self._dels = take._dels^
        if self._moves:
            self._moves.value()[] += 1

    fn __del__(deinit self):
        """Destroy the Observable and increment the del count."""
        if self._dels:
            self._dels.value()[] += 1


# ===----------------------------------------------------------------------=== #
# ConfigureTrivial
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ConfigureTrivial[
    *,
    del_is_trivial: Bool = False,
    copyinit_is_trivial: Bool = False,
    moveinit_is_trivial: Bool = False,
](Copyable):
    """Testing type configurable conditional triviality.

    Parameters:
        del_is_trivial: Whether the destructor is trivial.
        copyinit_is_trivial: Whether the copy initializer is trivial.
        moveinit_is_trivial: Whether the move initializer is trivial.
    """

    comptime __del__is_trivial = Self.del_is_trivial
    comptime __copyinit__is_trivial = Self.copyinit_is_trivial
    comptime __moveinit__is_trivial = Self.moveinit_is_trivial


# ===----------------------------------------------------------------------=== #
# NonMovable
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct NonMovable:
    """A non-movable type."""

    pass
