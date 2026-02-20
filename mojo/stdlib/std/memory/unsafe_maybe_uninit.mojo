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

from builtin.rebind import downcast
from os import abort
from memory import memset_zero


struct UnsafeMaybeUninit[T: AnyType](Copyable, Defaultable):
    """A wrapper type to represent memory that may or may not be initialized.

    `UnsafeMaybeUninit[T]` is a container for memory that may or may not
    be initialized. It is useful for dealing with uninitialized memory in a way
    that explicitly indicates to the compiler that the value inside might not be
    valid yet.

    For types with validity invariants, using uninitialized memory can cause
    undefined behavior.

    ## Important Safety Notes

    - **The destructor is a no-op**: `UnsafeMaybeUninit` never calls the
      destructor of `T`. If the memory was initialized, you **must**
      call `unsafe_assume_init_destroy()` before the memory is deallocated to
      properly clean up the value.

    - **Moving/copying behavior**: When you move or copy an
      `UnsafeMaybeUninit[T]`, only the raw bits are transferred. This
      operation does **not** invoke `T`'s move constructor or copy constructor.
      It is a simple bitwise copy of the underlying memory. This means:
      - Moving an `UnsafeMaybeUninit[T]` moves the bits, not the value
      - Copying an `UnsafeMaybeUninit[T]` copies the bits, not the value
      - No constructors or destructors are called during these operations

    - **Manual state tracking**: Every method in this struct is unsafe. You must
      track whether the memory is initialized or uninitialized at all times.
      Calling a method that assumes the memory is initialized (like
      `unsafe_assume_init_ref()`) when it is not will result in undefined
      behavior.

    - **Validity requirements**: `UnsafeMaybeUninit[T]` has no validity
      requirements, any bit pattern is valid. However, once you call
      `unsafe_assume_init_ref()`, the contained value must satisfy `T`'s
      validity requirements.

    Parameters:
        T: The type of the element to store.
    """

    comptime __del__is_trivial = True
    comptime __moveinit__is_trivial = _is_trivially_movable[Self.T]()
    comptime __copyinit__is_trivial = _is_trivially_copyable[Self.T]()

    comptime _mlir_type = __mlir_type[`!pop.array<1, `, Self.T, `>`]

    var _array: Self._mlir_type

    @always_inline
    fn __init__(out self):
        """The memory is now considered uninitialized."""
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    fn __init__[
        MovableType: Movable
    ](out self: UnsafeMaybeUninit[MovableType], var value: MovableType):
        """Create an `UnsafeMaybeUninit` in an initialized state.

        Parameters:
            MovableType: The type of the element to store.

        Args:
            value: The value to initialize the memory with.
        """
        self = UnsafeMaybeUninit[MovableType]()
        self.init_from(value^)

    @staticmethod
    @always_inline
    fn zeroed() -> Self:
        """Create an `UnsafeMaybeUninit` in an uninitialized state, with the memory set to all 0 bytes.

        It depends on `T` whether zeroed memory makes for proper initialization.
        For example, `UnsafeMaybeUninit[Int].zeroed()` is initialized,
        but `MaybeUninit[String].zeroed()` is not.

        Returns:
            An `UnsafeMaybeUninit` with the memory set to all 0 bytes.
        """
        var result = Self()
        memset_zero(UnsafePointer(to=result), 1)
        return result^

    fn __copyinit__(out self, copy: Self):
        """Copies the raw bits from another `UnsafeMaybeUninit` instance.

        This performs a bitwise copy of the underlying memory without invoking
        `T`'s copy constructor. For `UnsafeMaybeUninit[T]` to be Copyable,
        the held value `T` must be trivially copyable.

        Args:
            copy: The instance to copy from.
        """
        comptime assert (
            conforms_to(Self.T, Copyable)
            and downcast[Self.T, Copyable].__copyinit__is_trivial
        )
        self._array = copy._array

    fn __moveinit__(out self, deinit take: Self):
        """Moves the raw bits from another `UnsafeMaybeUninit` instance.

        This performs a bitwise move of the underlying memory without invoking
        `T`'s move constructor. For `UnsafeMaybeUninit[T]` to be Movable,
        the held value `T` must be trivially movable.

        Args:
            take: The value to move from.
        """
        comptime assert (
            conforms_to(Self.T, Movable)
            and downcast[Self.T, Movable].__moveinit__is_trivial
        )
        self._array = take._array

    @always_inline
    fn init_from[
        MovableType: Movable
    ](mut self: UnsafeMaybeUninit[MovableType], var value: MovableType):
        """Initialize this memory with the given `value`.

        This overwrite any previous value without destroying it.
        This means, if an previous `T` existed in the memory, that old instance
        will not be destoryed potentially leading to memory leaks.

        Parameters:
            MovableType: The type object to move.

        Args:
            value: The value to store in memory.
        """
        self.unsafe_ptr().init_pointee_move(value^)

    @always_inline
    fn unsafe_assume_init_ref(ref self) -> ref[self._array] Self.T:
        """Returns a reference to the internal value.

        Calling this method assumes that the memory is initialized.

        Returns:
            A reference to the internal value.
        """
        return self.unsafe_ptr()[]

    @always_inline
    fn unsafe_assume_init_take[
        U: Movable, //
    ](mut self: UnsafeMaybeUninit[U]) -> U:
        """Takes ownership of the internal value.

        Calling this method assumes that the memory is initialized. The value
        is moved out of the `UnsafeMaybeUninit` and returned to the caller.
        After this call, the memory is considered uninitialized.

        Parameters:
            U: The element type, which must be Movable.

        Returns:
            The initialized value that was stored in this container.
        """
        return self.unsafe_ptr().take_pointee()

    @always_inline
    fn unsafe_ptr(
        ref self,
    ) -> UnsafePointer[Self.T, origin_of(self._array)]:
        """Get a pointer to the underlying element.

        Note that this method does not assumes that the memory is initialized
        or not. It can always be called.

        Returns:
            A pointer to the underlying element.
        """
        return UnsafePointer(to=self._array).bitcast[Self.T]()

    @always_inline
    fn unsafe_assume_init_destroy[
        D: ImplicitlyDestructible
    ](mut self: UnsafeMaybeUninit[D]):
        """Runs the destructor of the internal value.

        Calling this method assumes that the memory is initialized.

        Parameters:
            D: An element type that is implicitly destructible.

        """
        self.unsafe_ptr().destroy_pointee()


@always_inline
fn _is_trivially_copyable[T: AnyType]() -> Bool:
    comptime if conforms_to(T, Copyable):
        return downcast[T, Copyable].__copyinit__is_trivial
    return False


@always_inline
fn _is_trivially_movable[T: AnyType]() -> Bool:
    comptime if conforms_to(T, Movable):
        return downcast[T, Movable].__moveinit__is_trivial
    return False
