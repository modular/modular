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
"""Implements `OwnedPointer`, a safe, single-ownership smart pointer.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import OwnedPointer
```
"""


@register_passable
struct OwnedPointer[T: AnyType]:
    """A safe, owning, smart pointer.

    This smart pointer is designed for cases where there is clear ownership
    of the underlying data, and restricts access to it through the origin
    system such that no more than one mutable alias for the underlying data
    may exist.

    For a comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

    Parameters:
        T: The type to be stored in the `OwnedPointer`.
    """

    var _inner: UnsafePointer[T, address_space = AddressSpace.GENERIC]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__[T: Movable](out self: OwnedPointer[T], var value: T):
        """Construct a new `OwnedPointer` by moving the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store. It is restricted to `Movable` here to allow efficient move construction.

        Args:
            value: The value to move into the `OwnedPointer`.
        """
        self._inner = UnsafePointer[T].alloc(1)
        self._inner.init_pointee_move(value^)

    fn __init__[T: Copyable](out self: OwnedPointer[T], *, copy_value: T):
        """Construct a new `OwnedPointer` by explicitly copying the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store, which must be
               `Copyable`.

        Args:
            copy_value: The value to explicitly copy into the `OwnedPointer`.
        """
        self._inner = UnsafePointer[T].alloc(1)
        self._inner.init_pointee_explicit_copy(copy_value)

    fn __init__[
        T: Copyable, U: NoneType = None
    ](out self: OwnedPointer[T], value: T):
        """Construct a new `OwnedPointer` by copying the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store.
            U: A dummy type parameter, to lower the selection priority of this ctor.

        Args:
            value: The value to copy into the `OwnedPointer`.
        """
        self._inner = UnsafePointer[T].alloc(1)
        self._inner.init_pointee_copy(value)

    fn __init__[
        T: Copyable
    ](out self: OwnedPointer[T], *, other: OwnedPointer[T]):
        """Construct a new `OwnedPointer` by explicitly copying the value from another `OwnedPointer`.

        Parameters:
            T: The type of the data to store.

        Args:
            other: The `OwnedPointer` to copy.
        """
        self = OwnedPointer[T](copy_value=other[])

    fn __init__(out self, *, unsafe_from_raw_pointer: UnsafePointer[T]):
        """Construct a new `OwnedPointer` by taking ownership of the provided `UnsafePointer`.

        Args:
            unsafe_from_raw_pointer: The `UnsafePointer` to take ownership of.

        ### Safety

        This function is unsafe as the provided `UnsafePointer` must be initialize with a single valid `T`
        initialliy allocated with this `OwnedPointer`'s backing allocator.
        This function is unsafe as other memory problems can arise such as a double-free if this function
        is called twice with the same pointer or a user manually deallocates the same data.

        After using this constructor, the `UnsafePointer` is assumed to be owned by this `OwnedPointer`.
        In particular, the destructor method will call `T.__del__` and `UnsafePointer.free`.
        """
        self._inner = unsafe_from_raw_pointer

    fn __del__(deinit self: OwnedPointer[T]):
        """Destroy the OwnedPointer[]."""
        self._inner.destroy_pointee()
        self._inner.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__(
        ref [AddressSpace.GENERIC]self,
    ) -> ref [self, AddressSpace.GENERIC] T:
        """Returns a reference to the pointers's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the `OwnedPointer`.
        """
        # This should have a widening conversion here that allows
        # the mutable ref that is always (potentially unsafely)
        # returned from UnsafePointer to be guarded behind the
        # aliasing guarantees of the origin system here.
        # All of the magic happens above in the function signature
        return self._inner[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """UNSAFE: returns the backing pointer for this `OwnedPointer`.

        Returns:
            An UnsafePointer to the backing allocation for this `OwnedPointer`.
        """
        return self._inner

    fn take[T: Movable](deinit self: OwnedPointer[T]) -> T:
        """Move the value within the `OwnedPointer` out of it, consuming the
        `OwnedPointer` in the process.

        Parameters:
            T: The type of the data backing this `OwnedPointer`. `take()` only exists for `T: Movable`
                since this consuming operation only makes sense for types that you want to avoid copying.
                For types that are `ImplicitlyCopyable` or `Copyable` but are not `Movable`, you can copy them through
                `__getitem__` as in `var v = some_ptr_var[]`.

        Returns:
            The data that is (was) backing the `OwnedPointer`.
        """
        var r = self._inner.take_pointee()
        self._inner.free()
        return r^

    fn steal_data(deinit self) -> UnsafePointer[T]:
        """Take ownership over the heap allocated pointer backing this
        `OwnedPointer`.

        ### Safety

        This function is not unsafe to call, as a memory leak is not
        considered unsafe.

        However, to avoid a memory leak, callers should ensure that the
        returned pointer is eventually deinitialized and deallocated.
        Failure to do so will leak memory.

        Returns:
            The pointer owned by this instance.
        """
        return self._inner
