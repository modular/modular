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
"""Implements `UnsafeMaybeUninit`, a wrapper for memory that may or may not be initialized."""

from std.builtin.rebind import downcast
from std.os import abort
from std.memory import (
    is_trivially_copyable,
    is_trivially_movable,
    unsafe_memset_zero,
)


struct UnsafeMaybeUninit[T: AnyType](
    Defaultable,
    ImplicitlyCopyable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
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
      call `unsafe_deinit()` before the memory is deallocated to
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
      `unsafe_assume_init()`) when it is not will result in undefined
      behavior.

    - **Validity requirements**: `UnsafeMaybeUninit[T]` has no validity
      requirements, any bit pattern is valid. However, once you call
      `unsafe_assume_init()`, the contained value must satisfy `T`'s
      validity requirements.

    Parameters:
        T: The type of the element to store.
    """

    comptime __del__is_trivial = True
    comptime __move_ctor_is_trivial = _is_trivially_movable[Self.T]()
    comptime __copy_ctor_is_trivial = _is_trivially_copyable[Self.T]()

    comptime _mlir_type = __mlir_type[`!pop.array<1, `, Self.T, `>`]

    var _array: Self._mlir_type

    @always_inline
    def __init__(out self):
        """The memory is now considered uninitialized."""
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def __init__(
        out self, var value: Self.T, /
    ) where conforms_to(Self.T, Movable):
        """Create an `UnsafeMaybeUninit` in an initialized state.

        Args:
            value: The value to initialize the memory with.
        """
        self = Self()
        self.unsafe_write(value^)

    @staticmethod
    @always_inline
    def zeroed() -> Self:
        """Create an `UnsafeMaybeUninit` in an uninitialized state, with the memory set to all 0 bytes.

        It depends on `T` whether zeroed memory makes for proper initialization.
        For example, `UnsafeMaybeUninit[Int].zeroed()` is initialized,
        but `MaybeUninit[String].zeroed()` is not.

        Returns:
            An `UnsafeMaybeUninit` with the memory set to all 0 bytes.
        """
        var result = Self()
        unsafe_memset_zero(Pointer(to=result), 1)
        return result^

    def __init__(out self, *, copy: Self):
        """Copies the raw bits from another `UnsafeMaybeUninit` instance.

        This performs a bitwise copy of the underlying memory without invoking
        `T`'s copy constructor. For `UnsafeMaybeUninit[T]` to be Copyable,
        the held value `T` must be trivially copyable.

        Args:
            copy: The instance to copy from.
        """
        comptime assert conforms_to(Self.T, Copyable)
        comptime assert is_trivially_copyable[Self.T]()
        self._array = copy._array

    def __init__(out self, *, deinit move: Self):
        """Moves the raw bits from another `UnsafeMaybeUninit` instance.

        This performs a bitwise move of the underlying memory without invoking
        `T`'s move constructor. For `UnsafeMaybeUninit[T]` to be Movable,
        the held value `T` must be trivially movable.

        Args:
            move: The value to move from.
        """
        comptime assert conforms_to(Self.T, Movable)
        comptime assert is_trivially_movable[Self.T]()
        self._array = move._array

    @always_inline
    def unsafe_write(
        mut self, var value: Self.T, /
    ) where conforms_to(Self.T, Movable):
        """Initialize this memory with the given `value`.

        This overwrite any previous value without destroying it.
        This means, if an previous `T` existed in the memory, that old instance
        will not be destroyed potentially leading to memory leaks.

        Args:
            value: The value to store in memory.

        Safety:

        - If the memory is already initialized, calling this leaks the
          previous value: its destructor never runs. Call `unsafe_deinit()`
          first if the previous value needs to be destroyed.
        """
        self.unsafe_ptr().unsafe_write(value^)

    @always_inline
    def unsafe_assume_init(
        deinit self,
    ) -> Self.T where conforms_to(Self.T, Movable):
        """Takes ownership of the contained value.

        Calling this method assumes that the memory is initialized. The
        value is moved out of the `UnsafeMaybeUninit` and returned to the
        caller. After this call, the memory is considered uninitialized.

        Returns:
            The initialized value that was stored in this container.

        Safety:

        - The memory must be initialized with a live `T` value. Calling this
          on uninitialized memory reads an invalid bit pattern as `T`, which
          is undefined behavior.
        """
        return self.unsafe_ptr().unsafe_take_pointee()

    @always_inline
    def unsafe_assume_init(ref self) -> ref[self] Self.T:
        """Returns a reference to the internal value.

        Calling this method assumes that the memory is initialized.

        Returns:
            A reference to the internal value.

        Safety:

        - The memory must be initialized with a live `T` value. Calling this
          on uninitialized memory produces a reference to an invalid bit
          pattern, which is undefined behavior if the reference is read.
        """
        return self.unsafe_ptr()[]

    @always_inline
    def unsafe_deinit(deinit self) where conforms_to(Self.T, Deinitable):
        """Destroys the contained value.

        Calling this method assumes that the memory is initialized. It runs
        `T`'s destructor on the contained value. After this call, the memory
        is considered uninitialized.

        Safety:

        - The memory must be initialized with a live `T` value. Calling this
          on uninitialized memory runs `T`'s destructor on an invalid bit
          pattern, which is undefined behavior.
        """
        self.unsafe_ptr().unsafe_deinit_pointee()

    @always_inline
    def unsafe_forget(deinit self):
        """Discards this `UnsafeMaybeUninit` without destroying its contents.

        Unlike `unsafe_deinit()`, this does not run `T`'s destructor. Use
        this when the memory is uninitialized, or when the contained value
        has already been disposed of some other way.

        Safety:

        - If the memory is initialized with a value that owns a resource
          (for example, an allocation), calling this leaks that resource:
          its destructor never runs. Call `unsafe_deinit()` instead if the
          value needs to be destroyed.
        """
        pass

    @always_inline
    def unsafe_ptr(
        ref self,
    ) -> Pointer[Self.T, origin_of(self)]:
        """Get a pointer to the underlying element.

        Note that this method does not assumes that the memory is initialized
        or not. It can always be called.

        Returns:
            A pointer to the underlying element.

        Safety:

        - The returned pointer may point to uninitialized memory. Reading
          through it before the memory is initialized is undefined behavior.
        """
        return (
            Pointer(to=self._array)
            .unsafe_bitcast[Self.T]()
            .unsafe_origin_cast[origin_of(self)]()
        )


@always_inline
def _is_trivially_copyable[T: AnyType]() -> Bool:
    comptime if conforms_to(T, Copyable):
        return is_trivially_copyable[T]()
    return False


@always_inline
def _is_trivially_movable[T: AnyType]() -> Bool:
    comptime if conforms_to(T, Movable):
        return is_trivially_movable[T]()
    return False
