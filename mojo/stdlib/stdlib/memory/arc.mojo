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
"""Reference-counted smart pointers.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import ArcPointer
```
"""

from os.atomic import Atomic, Consistency, fence
from sys.info import size_of


struct _ArcPointerInner[T: Movable]:
    var refcount: Atomic[DType.uint64]
    var payload: T

    fn __init__(out self, var value: T):
        """Create an initialized instance of this with a refcount of 1."""
        self.refcount = Atomic(UInt64(1))
        self.payload = value^

    fn add_ref(mut self):
        """Atomically increment the refcount."""

        # `MONOTONIC` is ok here since this ArcPointer is currently being copied
        # from an existing ArcPointer inside of __copyinit__. This means any
        # other ArcPointer in different threads running their destructors will
        # not see a refcount of 0 and will not delete the shared data.
        #
        # This is further explained in the [boost documentation]
        # (https://www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        _ = self.refcount.fetch_add[ordering = Consistency.MONOTONIC](1)

    fn drop_ref(mut self) -> Bool:
        """Atomically decrement the refcount and return true if the result
        hits zero."""

        # `RELEASE` is needed to ensure that all data access happens before
        # decreasing the refcount. `ACQUIRE_RELEASE` is not needed since we
        # don't need the guarantees of `ACQUIRE` on the load portion of
        # fetch_sub if the recount does not reach zero.
        if self.refcount.fetch_sub[ordering = Consistency.RELEASE](1) != 1:
            return False

        # However, if the refcount results in zero, this `ACQUIRE` fence is
        # needed to synchronize with the `fetch_sub[RELEASE]` above, ensuring
        # that use of data happens before the fence and therefore before the
        # deletion of the data.
        fence[ordering = Consistency.ACQUIRE]()
        return True


@register_passable
struct ArcPointer[T: Movable](Identifiable, ImplicitlyCopyable, Movable):
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
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

    Examples:

    ```mojo
    from memory import ArcPointer
    var p = ArcPointer(4)
    var p2 = p
    p2[]=3
    print(3 == p[])
    ```

    Parameters:
        T: The type of the stored value.
    """

    alias _inner_type = _ArcPointerInner[T]
    var _inner: UnsafePointer[Self._inner_type]

    fn __init__(out self, var value: T):
        """Construct a new thread-safe, reference-counted smart pointer,
        and move the value into heap memory managed by the new pointer.

        Args:
            value: The value to manage.
        """
        self._inner = UnsafePointer[Self._inner_type].alloc(1)
        # Cannot use init_pointee_move as _ArcPointerInner isn't movable.
        __get_address_as_uninit_lvalue(self._inner.address) = Self._inner_type(
            value^
        )

    fn __init__(out self, *, unsafe_from_raw_pointer: UnsafePointer[T]):
        """Constructs an `ArcPointer` from a raw pointer.

        Args:
            unsafe_from_raw_pointer: A raw pointer previously returned from `ArcPointer.steal_data`.

        ### Safety

        The `unsafe_from_raw_pointer` argument *must* have been previously returned by a call
        to `ArcPointer.steal_data`. Any other pointer may result in undefined behaviour.

        ### Example

        ```mojo
        from memory import ArcPointer

        var initial_arc = ArcPointer[Int](42)
        var raw_ptr = initial_arc^.steal_data()

        # The following will ensure the data is properly destroyed and deallocated.
        var restored_arc = ArcPointer(unsafe_from_raw_pointer=raw_ptr)
        ```
        """
        var pointer_to_payload = unsafe_from_raw_pointer.bitcast[Byte]()

        # Calculate the offset to the beginning of the `_ArcPointerInner` allocation.
        var pointer_to_inner = (
            pointer_to_payload - size_of[__type_of(self._inner[].refcount)]()
        )
        self._inner = pointer_to_inner.bitcast[Self._inner_type]()

    fn __copyinit__(out self, existing: Self):
        """Copy an existing reference. Increment the refcount to the object.

        Args:
            existing: The existing reference.
        """
        # Order here does not matter since `existing` can't be destroyed until
        # sometime after we return.
        existing._inner[].add_ref()
        self._inner = existing._inner

    @no_inline
    fn __del__(deinit self):
        """Delete the smart pointer.

        Decrement the reference count for the stored value. If there are no more
        references, delete the object and free its memory."""
        if self._inner[].drop_ref():
            # Call inner destructor, then free the memory.
            self._inner.destroy_pointee()
            self._inner.free()

    # FIXME: The origin returned for this is currently self origin, which
    # keeps the ArcPointer object alive as long as there are references into it.  That
    # said, this isn't really the right modeling, we need hierarchical origins
    # to model the mutability and invalidation of the returned reference
    # correctly.
    fn __getitem__[
        self_life: ImmutableOrigin
    ](ref [self_life]self) -> ref [MutableOrigin.cast_from[self_life]] T:
        """Returns a mutable reference to the managed value.

        Parameters:
            self_life: The origin of self.

        Returns:
            A reference to the managed value.
        """
        return self._inner[].payload

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The `UnsafePointer` to the pointee.
        """
        # TODO: consider removing this method.
        return UnsafePointer(to=self._inner[].payload)

    fn count(self) -> UInt64:
        """Count the amount of current references.

        Returns:
            The current amount of references to the pointee.
        """
        # TODO: this should use `load[MONOTONIC]()` once load supports memory
        # orderings.
        #
        # MONOTONIC is okay here - reading refcount simply needs to be atomic.
        # No synchronization is needed as this is not attempting to free the
        # shared data and it is not possible for the data to be freed until
        # this ArcPointer is destroyed.
        return self._inner[].refcount.fetch_sub[
            ordering = Consistency.MONOTONIC
        ](0)

    fn steal_data(deinit self) -> UnsafePointer[T]:
        """Consume this `ArcPointer`, returning a raw pointer to the underlying data.

        Returns:
            An `UnsafePointer` to the underlying `T` value.

        ### Safety

        To avoid leaking memory, this pointer must be converted back to an `ArcPointer`
        using `ArcPointer(unsafe_from_raw_pointer=ptr)`.
        The returned pointer is not guaranteed to point to the beginning of the backing allocation,
        meaning calling `UnsafePointer.free` may result in undefined behavior.
        """
        return UnsafePointer(to=self._inner[].payload)

    fn __is__(self, rhs: Self) -> Bool:
        """Returns True if the two `ArcPointer` instances point at the same
        object.

        Args:
            rhs: The other `ArcPointer`.

        Returns:
            True if the two `ArcPointers` instances point at the same object and
            False otherwise.
        """
        return self._inner == rhs._inner
