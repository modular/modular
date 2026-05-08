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
"""Reference-counted smart pointers.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import ArcPointer
```
"""

from std.atomic import Atomic, Ordering, fence
from std.format._utils import (
    Repr,
    FormatStruct,
    TypeNames,
)
from std.hashlib.hasher import Hasher
from std.memory.unsafe_maybe_uninit import UnsafeMaybeUninit
from std.reflection import reflect
from std.memory import alloc, free, Layout


@doc_hidden
struct _ArcPointerInner[T: Movable & ImplicitlyDestructible]:
    """
    The backing _shared_ piece of an ArcPointer.
    Referenced by all Arc and Weak for a given value.
    Carries the atomic refcounts and the T itself.
    """

    var strong: Atomic[DType.uint64]
    var weak: Atomic[DType.uint64]
    var payload: UnsafeMaybeUninit[Self.T]

    @doc_hidden
    def __init__(out self, var value: Self.T):
        """Create an initialized instance with strong=1 and weak=1.

        The weak counter starts at 1 to represent the implicit weak
        reference shared by all strong pointers.
        """
        self.strong = Atomic(UInt64(1))
        self.weak = Atomic(UInt64(1))
        self.payload = UnsafeMaybeUninit[Self.T](value^)

    def add_strong(mut self):
        """Atomically increment the strong refcount."""

        # `MONOTONIC` is ok here since this ArcPointer is currently being copied
        # from an existing ArcPointer inside of copy ctor. This means any
        # other ArcPointer in different threads running their destructors will
        # not see a refcount of 0 and will not delete the shared data.
        #
        # This is further explained in the [boost documentation]
        # (https://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        _ = self.strong.fetch_add[ordering=Ordering.RELAXED](1)

    def drop_strong(mut self) -> Bool:
        """Atomically decrement the strong refcount and return true if the
        result hits zero."""

        # `RELEASE` is needed to ensure that all data access happens before
        # decreasing the refcount. `ACQUIRE_RELEASE` is not needed since we
        # don't need the guarantees of `ACQUIRE` on the load portion of
        # fetch_sub if the refcount does not reach zero.
        if self.strong.fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return False

        # However, if the refcount results in zero, this `ACQUIRE` fence is
        # needed to synchronize with the `fetch_sub[RELEASE]` above, ensuring
        # that use of data happens before the fence and therefore before the
        # deletion of the data.
        fence[ordering=Ordering.ACQUIRE]()
        return True

    def try_add_strong(mut self) -> Bool:
        """Increment the strong refcount only if it is not already zero.

        Used by `WeakPointer.try_upgrade` to refuse resurrecting a destroyed payload.
        """
        var cur = self.strong.load[ordering=Ordering.RELAXED]()
        while cur != 0:
            # On failure, `compare_exchange` rewrites `cur` with the observed
            # value, so the next iteration retries against the fresh count.
            if self.strong.compare_exchange[
                success_ordering=Ordering.ACQUIRE,
                failure_ordering=Ordering.RELAXED,
            ](cur, cur + 1):
                return True
        return False

    def strong_count(self) -> UInt64:
        """Return the current strong refcount."""
        return self.strong.load[ordering=Ordering.RELAXED]()

    def add_weak(mut self):
        """Atomically increment the weak refcount."""
        _ = self.weak.fetch_add[ordering=Ordering.RELAXED](1)

    def drop_weak(mut self) -> Bool:
        """Atomically decrement the weak refcount and return true if the
        result hits zero."""
        if self.weak.fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return False
        fence[ordering=Ordering.ACQUIRE]()
        return True

    def weak_count_with_implicit(self) -> UInt64:
        """Return the raw weak counter, including the implicit +1 held
        collectively by all strong references."""
        return self.weak.load[ordering=Ordering.RELAXED]()

    def destroy_payload(mut self):
        """Run the destructor of the held value. Caller must ensure this
        runs exactly once, on the last strong drop."""
        self.payload.unsafe_assume_init_destroy()

    def payload_ref(ref self) -> ref[self.payload._array] Self.T:
        """Return a reference to the (assumed-initialized) payload."""
        return self.payload.unsafe_assume_init_ref()


struct ArcPointer[T: Movable & ImplicitlyDestructible](
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    Identifiable,
    ImplicitlyCopyable,
    RegisterPassable,
    Writable where conforms_to(T, Writable),
):
    """Atomic reference-counted pointer.

    This smart pointer owns an instance of `T` indirectly managed on the heap.
    This pointer is copyable, including across threads, maintaining a reference
    count to the underlying data.

    When you initialize an `ArcPointer` with a value, it allocates memory and
    moves the value into the allocated memory. Copying an instance of an
    `ArcPointer` increments the reference count. Destroying an instance
    decrements the reference count. When the reference count reaches zero,
    `ArcPointer` destroys the value and frees its memory.

    This pointer itself is thread-safe using atomic accesses to reference count
    the underlying data, but references returned to the underlying data are not
    thread-safe.

    Subscripting an `ArcPointer` (`ptr[]`) returns a mutable reference to the
    stored value. This is the only safe way to access the stored value. Other
    methods, such as using the `unsafe_ptr()` method to retrieve an unsafe
    pointer to the stored value, or accessing the private fields of an
    `ArcPointer`, are unsafe and may result in memory errors.

    For a comparison with other pointer types, see [Intro to
    pointers](/docs/manual/pointers/) in the Mojo Manual.

    Examples:

    ```mojo
    from std.memory import ArcPointer
    var p = ArcPointer(4)
    var p2 = p
    p2[]=3
    print(3 == p[])
    ```

    Parameters:
        T: The type of the stored value.
    """

    comptime WeakPointer = WeakPointer[Self.T]
    """Convenience alias: `WeakPointer[T]` for this `ArcPointer[T]`."""

    comptime _inner_type = _ArcPointerInner[Self.T]
    var _inner: UnsafePointer[Self._inner_type, MutExternalOrigin]

    def __init__(out self, var value: Self.T):
        """Construct a new thread-safe, reference-counted smart pointer,
        and move the value into heap memory managed by the new pointer.

        Args:
            value: The value to manage.
        """
        self._inner = alloc(Layout[Self._inner_type].single())
        # Cannot use init_pointee_move as _ArcPointerInner isn't movable.
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(
            value^
        )

    def __init__(
        out self,
        *,
        unsafe_from_raw_pointer: UnsafePointer[Self.T, MutExternalOrigin],
    ):
        """Constructs an `ArcPointer` from a raw pointer.

        Args:
            unsafe_from_raw_pointer: A raw pointer previously returned from `ArcPointer.steal_data`.

        **Safety:**

        The `unsafe_from_raw_pointer` argument *must* have been previously returned by a call
        to `ArcPointer.steal_data`. Any other pointer may result in undefined behaviour.

        **Example:**

        ```mojo
        from std.memory import ArcPointer

        var initial_arc = ArcPointer[Int](42)
        var raw_ptr = initial_arc^.steal_data()

        # The following will ensure the data is properly destroyed and deallocated.
        var restored_arc = ArcPointer(unsafe_from_raw_pointer=raw_ptr)
        ```
        """
        var pointer_to_payload = unsafe_from_raw_pointer.bitcast[Byte]()
        comptime payload_offset = reflect[Self._inner_type].field_offset[
            name="payload"
        ]()
        var pointer_to_inner = pointer_to_payload - payload_offset
        self._inner = pointer_to_inner.bitcast[Self._inner_type]()

    def __init__(out self, *, copy: Self):
        """Copy an existing reference. Increment the refcount to the object.

        Args:
            copy: The existing reference.
        """
        # Order here does not matter since `copy` can't be destroyed until
        # sometime after we return.
        copy._inner[].add_strong()
        self._inner = copy._inner

    def __init__(
        out self,
        *,
        _inner: UnsafePointer[Self._inner_type, MutExternalOrigin],
    ):
        """Internal: construct from an already-incremented inner pointer.

        Args:
            _inner: The inner control-block pointer. The strong refcount
                must already account for this new `ArcPointer`.
        """
        self._inner = _inner

    @no_inline
    def __del__(deinit self):
        """Delete the smart pointer.

        Decrement the reference count for the stored value. If there are no more
        references, delete the object and free its memory."""
        if not self._inner[].drop_strong():
            return

        # Last strong reference: destroy the payload. The control block
        # itself stays alive while any Weak references still hold the
        # implicit-weak refcount.
        self._inner[].destroy_payload()

        # Drop the implicit weak reference held collectively by all strong
        # pointers. If we are also the last weak, free the allocation.
        if self._inner[].drop_weak():
            free(self._inner, {count = 1})

    # FIXME: The origin returned for this is currently self origin, which
    # keeps the ArcPointer object alive as long as there are references into it.  That
    # said, this isn't really the right modeling, we need indirect origins
    # to model the mutability and invalidation of the returned reference
    # correctly.
    def __getitem__[
        self_life: ImmutOrigin
    ](ref[self_life] self) -> ref[self_life.unsafe_mut_cast[True]()] Self.T:
        """Returns a mutable reference to the managed value.

        Parameters:
            self_life: The origin of self.

        Returns:
            A reference to the managed value.
        """
        return self._inner[].payload_ref()

    def unsafe_ptr[
        mut: Bool,
        origin: Origin[mut=mut],
        //,
    ](ref[origin] self) -> UnsafePointer[Self.T, origin]:
        """Retrieves a pointer to the underlying memory.

        Parameters:
            mut: Whether the pointer is mutable.
            origin: The origin of the pointer.

        Returns:
            An `UnsafePointer` to the pointee.
        """
        # TODO: consider removing this method.
        return (
            UnsafePointer(to=self._inner[].payload_ref())
            .mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )

    def count(self) -> UInt64:
        """Returns the current strong reference count.

        Returns:
            The current number of strong references to the pointee.
        """
        # MONOTONIC is okay here - reading refcount simply needs to be atomic.
        # No synchronization is needed as this is not attempting to free the
        # shared data and it is not possible for the data to be freed until
        # this ArcPointer is destroyed.
        return self._inner[].strong_count()

    def weak_count(self) -> UInt64:
        """Returns the current number of `Weak` pointers, excluding the
        implicit weak reference held by all strong pointers.

        Returns:
            The number of outstanding `Weak` pointers to this allocation.
        """

        # Don't include the implicit weak from the strong pointers.
        # Don't need to check if there is one, since this call
        # is from a strong and there _must_ be exactly one implicit.
        return self._inner[].weak_count_with_implicit() - 1

    def steal_data(deinit self) -> UnsafePointer[Self.T, MutExternalOrigin]:
        """Consume this `ArcPointer`, returning a raw pointer to the underlying data.

        Returns:
            An `UnsafePointer` to the underlying `T` value.

        **Safety:**

        To avoid leaking memory, this pointer must be converted back to an `ArcPointer`
        using `ArcPointer(unsafe_from_raw_pointer=ptr)`.
        The returned pointer is not guaranteed to point to the beginning of the backing allocation,
        meaning calling `UnsafePointer.free` may result in undefined behavior.
        """
        return UnsafePointer(to=self._inner[].payload_ref())

    def __is__(self, rhs: Self) -> Bool:
        """Returns True if the two `ArcPointer` instances point at the same
        object.

        Args:
            rhs: The other `ArcPointer`.

        Returns:
            True if the two `ArcPointers` instances point at the same object and
            False otherwise.
        """
        return self._inner == rhs._inner

    def __eq__(self, rhs: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Returns True if the two `ArcPointer` instances hold equal values.

        Delegates to the underlying value's `__eq__` method, so two
        `ArcPointer` instances holding equal values compare equal even if
        they are separate allocations. This is consistent with `__hash__`.

        Args:
            rhs: The other `ArcPointer`.

        Returns:
            True if the managed values are equal, False otherwise.
        """
        return self[] == rhs[]

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Hash the managed value.

        Delegates to the underlying value's `__hash__` method, so two
        `ArcPointer` instances holding equal values produce the same hash.
        This is consistent with `__eq__`.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance to update.
        """
        self[].__hash__(hasher)

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Formats this pointer's value to the provided Writer.

        Args:
            writer: The object to write to.

        Constraints:
            T must conform to Writable.
        """
        self[].write_to(writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write the string representation of the `ArcPointer`.

        Args:
            writer: The object to write to.

        Constraints:
            T must conform to Writable.
        """
        FormatStruct(writer, "ArcPointer").params(
            TypeNames[Self.T](),
        ).fields(Repr(self[]))


struct WeakPointer[T: Movable & ImplicitlyDestructible](
    ImplicitlyCopyable, RegisterPassable
):
    """Non-owning atomic reference to an `ArcPointer`'s allocation.

    A `WeakPointer` may _not_ be directly subscripted to dereference
    to the value it conceptually points to.

    Instead, a `WeakPointer` may be `.try_upgrade()`ed into an `ArcPointer[T]` if there is
    at least one `ArcPointer[T]` for the value still live.

    This may be used for self-referential structures, such as doubly
    linked lists or trees with parent refs, to avoid creating ownership
    cycles that would leak if dropped.

    If all `ArcPointer[T]` to a value are dropped but `WeakPointer[T]` remain,
    the value will be destroyed. After this point, `.try_upgrade()`ing a `WeakPointer[T]`
    to that value will always return `None`.

    Parameters:
        T: The type of the stored value.
    """

    comptime _inner_type = _ArcPointerInner[Self.T]
    comptime _inner_ptr_type = UnsafePointer[
        Self._inner_type, MutExternalOrigin
    ]
    var _inner: Optional[Self._inner_ptr_type]

    def __init__(
        out self,
    ):
        """
        Create a null WeakPointer, with _no_ inner.
        This can be used for transient construction states for
        self-referential structures.
        """
        self._inner = Optional[Self._inner_ptr_type]()

    def __init__(
        out self,
        *,
        downgrade: ArcPointer[Self.T],
    ):
        """Creates a new `Weak` pointer from an associated `ArcPointer`.

        Args:
            downgrade: The value that this creates a WeakPointer referring to.

        Returns:
            A new `Weak` pointer sharing this allocation.
        """
        downgrade._inner[].add_weak()
        self._inner = downgrade._inner

    @doc_hidden
    def __init__(
        out self,
        *,
        _inner: Self._inner_ptr_type,
    ):
        """Internal: construct from an already-incremented inner pointer.

        Args:
            _inner: The inner control-block pointer. The weak refcount
                must already account for this new `WeakPointer`.
        """
        self._inner = _inner

    def __init__(out self, *, copy: Self):
        """Increment the weak count and share the allocation.

        Args:
            copy: The existing `WeakPointer` to share an allocation with.
        """
        if copy._inner:
            copy._inner.unsafe_value()[].add_weak()
        self._inner = copy._inner

    @no_inline
    def __del__(deinit self):
        """Decrement the weak count and free the allocation if last."""
        if self._inner and self._inner.unsafe_value()[].drop_weak():
            free(self._inner.unsafe_value(), {count = 1})

    def try_upgrade(self) -> Optional[ArcPointer[Self.T]]:
        """Attempts to obtain a strong reference.

        Returns:
            An `ArcPointer` sharing the allocation, or `None` if the
            payload has already been destroyed (strong count reached 0).
        """
        if self._inner and self._inner.unsafe_value()[].try_add_strong():
            return {ArcPointer[Self.T](_inner=self._inner.unsafe_value())}
        return Optional[ArcPointer[Self.T]]()

    def strong_count(self) -> UInt64:
        """Returns the current strong count, or 0 if the payload is gone.

        Returns:
            The current number of strong references to the allocation.
        """
        if self._inner:
            return self._inner.unsafe_value()[].strong_count()
        else:
            return 0

    def weak_count(self) -> UInt64:
        """Returns an approximate count of `WeakPointer`s to this shared value.

        This can be off by one in the presence of concurrent activity.

        Returns:
            The approximate number of outstanding `WeakPointer`s.
        """

        if self._inner:
            var w = self._inner.unsafe_value()[].weak_count_with_implicit()
            if self._inner.unsafe_value()[].strong_count() == 0:
                return w
            # If there are any strong remaining, we don't want to
            # include the implicit weak in the returned count.
            return w - 1
        else:
            return 0
