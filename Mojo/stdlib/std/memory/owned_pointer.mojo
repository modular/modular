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
"""Implements `OwnedPointer`, a safe, single-ownership smart pointer.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import OwnedPointer
```
"""

from std.format._utils import (
    Repr,
    FormatStruct,
    TypeNames,
)
from std.memory.alloc import (
    Allocation,
    ThinAllocation,
    alloc,
    dealloc,
    Layout,
)
from std.utils._nicheable import _FieldNicheable


struct OwnedPointer[T: AnyType](
    Deinitable where conforms_to(T, Deinitable) else (
        "Use `deinit_with()` or `into_inner()` (for a `Movable` `T`) to"
        " consume an `OwnedPointer` whose element type is not `Deinitable`"
    ),
    RegisterPassable,
    Writable where conforms_to(T, Writable),
    _FieldNicheable,
):
    """A safe, owning, smart pointer.

    This smart pointer is designed for cases where there is clear ownership
    of the underlying data, and restricts access to it through the origin
    system such that no more than one mutable alias for the underlying data
    may exist.

    For a comparison with other pointer types, see [Intro to
    pointers](/docs/manual/pointers/) in the Mojo Manual.

    Parameters:
        T: The type to be stored in the `OwnedPointer`. When `T` is not
            `Deinitable`, the `OwnedPointer` has no implicit
            destructor and must be consumed with `deinit_with()` or
            `into_inner()` (for a `Movable` `T`).
    """

    var _inner: ThinAllocation[Self.T]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(
        out self, var value: Self.T, /
    ) where conforms_to(Self.T, Movable):
        """Construct a new `OwnedPointer` by moving the passed value into a new backing allocation.

        Args:
            value: The value to move into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[Self.T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(value^)

    def __init__(out self, *, var init_with: Some[def() -> Self.T]):
        """Construct a new `OwnedPointer` holding the value returned by `init_with`.

        The value is constructed directly into the new backing allocation
        rather than being constructed elsewhere and then moved, so `T` does
        not need to be `Movable`.

        ```mojo
        from std.memory import OwnedPointer

        @fieldwise_init
        struct Pinned(not Movable):
            var value: Int

        var ptr = OwnedPointer(init_with=lambda () -> Pinned: Pinned(7))
        print(ptr[].value)  # => 7
        ```

        Args:
            init_with: A function that constructs and returns the value to
                store in the `OwnedPointer`.
        """
        self._inner = alloc(Layout[Self.T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(init_with=init_with)

    @deprecated(
        "This `OwnedPointer` initializer is deprecated. Use the by value or"
        " in-place initializers."
    )
    def __init__(
        out self, *, copy_value: Self.T
    ) where conforms_to(Self.T, Copyable):
        """Construct a new `OwnedPointer` by explicitly copying the passed value into a new backing allocation.

        Args:
            copy_value: The value to explicitly copy into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[Self.T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(copy=copy_value)

    # Implicitly copyable lvalues would otherwise pick this borrowing overload
    # over the by-value one, so exclude them to keep them off the deprecation.
    @deprecated(
        "This `OwnedPointer` initializer is deprecated. Use the by value or"
        " in-place initializers."
    )
    def __init__[
        _T: Copyable, U: NoneType = None
    ](out self: OwnedPointer[_T], value: _T) where not conforms_to(
        _T, ImplicitlyCopyable
    ):
        """Construct a new `OwnedPointer` by copying the passed value into a new backing allocation.

        Parameters:
            _T: The type of the data to store.
            U: A dummy type parameter, to lower the selection priority of this ctor.

        Args:
            value: The value to copy into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[_T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(copy=value)

    def __init__(out self, *, other: Self) where conforms_to(Self.T, Copyable):
        """Construct a new `OwnedPointer` by explicitly copying the value from another `OwnedPointer`.

        Args:
            other: The `OwnedPointer` to copy.
        """
        self = {init_with = lambda () -> Self.T: other[].copy()}

    def __init__(
        out self,
        *,
        unsafe_from_raw_pointer: Pointer[Self.T, MutUntrackedOrigin],
    ):
        """Construct a new `OwnedPointer` by taking ownership of the provided `Pointer`.

        Args:
            unsafe_from_raw_pointer: The `Pointer` to take ownership of.

        Safety:

        This function is unsafe as the provided `Pointer` must be initialize with a single valid `T`
        initially allocated with this `OwnedPointer`'s backing allocator.
        This function is unsafe as other memory problems can arise such as a double-free if this function
        is called twice with the same pointer or a user manually deallocates the same data.

        After using this constructor, the `Pointer` is assumed to be owned by this `OwnedPointer`.
        In particular, the destructor method will call `T.__deinit__` and `dealloc`.
        """
        self._inner = ThinAllocation(unsafe_owned_ptr=unsafe_from_raw_pointer)

    def __init__(out self, *, unsafe_from_opaque_pointer: MutOpaquePointer[_]):
        """Construct a new `OwnedPointer` by taking ownership of the provided `Pointer`.

        Args:
            unsafe_from_opaque_pointer: The `OpaquePointer` to take ownership of.

        Safety:

        This function is unsafe as the provided `OpaquePointer` must be initialize with a single valid `T`
        initially allocated with this `OwnedPointer`'s backing allocator.
        This function is unsafe as other memory problems can arise such as a double-free if this function
        is called twice with the same pointer or a user manually deallocates the same data.

        After using this constructor, the `Pointer` is assumed to be owned by this `OwnedPointer`.
        In particular, the destructor method will call `T.__deinit__` and `dealloc`.
        """
        var ptr = unsafe_from_opaque_pointer.unsafe_bitcast[Self.T]()
        self = Self(
            unsafe_from_raw_pointer=ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        )

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        """Destroy the `OwnedPointer`, running the destructor of its value.

        Constraints:
            `T` must be `Deinitable`. When it is not, the
            `OwnedPointer` has no implicit destructor and must be consumed
            with `deinit_with()` or `into_inner()` (for a `Movable` `T`).
        """
        self._inner.unsafe_ptr().unsafe_deinit_pointee()
        dealloc(self._inner^.unsafe_with_layout(Layout[Self.T].single()))

    def deinit_with(deinit self, deinit_func: Some[def(var Self.T)], /):
        """Consumes this `OwnedPointer`, passing its value to `deinit_func` to
        destroy it.

        The value is destroyed in place in its backing allocation, so `T`
        does not need to be `Movable` or `Deinitable`.

        Args:
            deinit_func: The function that takes ownership of the value to
                destroy it.

        Examples:

        ```mojo
        from std.memory import OwnedPointer

        @fieldwise_init
        struct Resource(not Deinitable, not Movable):
            var id: Int

            def close(deinit self):
                pass

        var p = OwnedPointer(init_with=lambda () -> Resource: Resource(1))
        p^.deinit_with(Resource.close)
        ```
        """
        self._inner.unsafe_ptr().unsafe_deinit_pointee_with(deinit_func)
        dealloc(self._inner^.unsafe_with_layout(Layout[Self.T].single()))

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref[AddressSpace.GENERIC] self,
    ) -> ref[
        origin_of(self)._get_owned_interior["value"], AddressSpace.GENERIC
    ] Self.T:
        """Returns a reference to the pointers's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the `OwnedPointer`.
        """
        return self._inner.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
            "value", origin_of(self)
        ]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def ptr[origin: Origin, //](ref[origin] self) -> Pointer[Self.T, origin]:
        """Returns a pointer to the `OwnedPointer`'s contents.

        Parameters:
            origin: The origin of the pointer.

        Returns:
            A pointer to the `OwnedPointer`'s contents.
        """
        return self._inner.unsafe_ptr().unsafe_origin_cast[origin]()

    @doc_hidden
    @deprecated(use=ptr)
    def unsafe_ptr[
        mut: Bool,
        origin: Origin[mut=mut],
        //,
    ](ref[origin] self) -> Pointer[Self.T, origin]:
        """Returns the backing pointer for this `OwnedPointer`.

        Parameters:
            mut: Whether the pointer is mutable.
            origin: The origin of the pointer.

        Returns:
            A pointer to the backing allocation for this `OwnedPointer`.
        """
        return self.ptr()

    def into_inner(deinit self) -> Self.T where conforms_to(Self.T, Movable):
        """Move the value within the `OwnedPointer` out of it, consuming the
        `OwnedPointer` in the process.

        Returns:
            The data that is (was) backing the `OwnedPointer`.
        """
        var r = self._inner.unsafe_ptr().unsafe_take_pointee()
        dealloc(self._inner^.unsafe_with_layout(Layout[Self.T].single()))
        return r^

    def unsafe_take_allocation(deinit self) -> Allocation[Self.T]:
        """Take ownership of the heap allocation backing this `OwnedPointer`.

        Returns:
            The `Allocation` that owns the backing storage.

        Safety:

        The pointee is handed over still initialized, and deallocating the
        storage does not run its destructor. Destroy it yourself before
        deallocating if `T` needs it.

        The returned `Allocation` is an explicitly destroyed handle, so the
        compiler requires the caller to consume it: pass it to `dealloc`, or
        take the raw pointer with `unsafe_leak()`.
        """
        return self._inner^.unsafe_with_layout(Layout[Self.T].single())

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Formats this pointer's value to the provided Writer.

        Args:
            writer: The object to write to.

        Constraints:
            `T` must conform to Writable.
        """
        self[].write_to(writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write the string representation of the `OwnedPointer`.

        Args:
            writer: The object to write to.

        Constraints:
            `T` must conform to Writable.
        """
        FormatStruct(writer, "OwnedPointer").params(
            TypeNames[Self.T](),
        ).fields(Repr(self[]))

    comptime _KgenNichedFieldName = "_inner".value
