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
"""Defines `Deinitable`, the trait for types with lifetime management through destructors.

This is a Mojo built-in, so you don't need to import it.
"""

from std.builtin.rebind import downcast


@stable(since="1.0")
trait Deinitable:
    """A trait for types that require lifetime management through destructors.

    The `Deinitable` trait is fundamental to Mojo's memory
    management system. It indicates that a type has a destructor that needs to
    be called when instances go out of scope. This is essential for types that
    own resources like memory, file handles, or other system resources that need
    proper cleanup.

    By default, all Mojo types implement `Deinitable`, unless they
    opt-in to required explicit named destructor methods using a
    `Deinitable where False`, or conditionally with
    `Deinitable where <cond>`.

    Key aspects:

    - Any type with an implicit `__deinit__()` destructor must implement this trait
    - The destructor (`__deinit__`) is called automatically when an instance's
      lifetime ends
    - Composition of types with implicit destructors automatically get an
      implicit destructor

    Example:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout, Allocation

    struct ResourceOwner(Deinitable):
        var allocation: Allocation[Int]

        def __init__(out self, size: Int):
            self.allocation = alloc(Layout[Int](count=size))

        def __deinit__(deinit self):
            # Clean up owned resources
            dealloc(self.allocation^)
    ```

    Best practices:

    - Implement this trait when your type owns resources that need cleanup
    - Ensure the destructor properly frees all owned resources
    - Consider using a `Deinitable where False` conformance on types
      that should never be deleted implicitly.
    - Use composition to automatically handle nested resource cleanup
    """

    @stable(since="1.0")
    def __deinit__(deinit self, /):
        """Destroys the instance and cleans up any owned resources.

        This method is called automatically when an instance's lifetime ends. It receives
        an owned value and should perform all necessary cleanup operations like:
        - Freeing allocated memory
        - Closing file handles
        - Releasing system resources
        - Cleaning up any other owned resources

        The instance is considered dead after this method completes, regardless of
        whether any explicit cleanup was performed.
        """
        ...

    comptime __del__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation of `__deinit__` is trivial.

    The implementation of `__deinit__` is considered to be trivial if:
    - The struct has a compiler-generated trivial destructor and all its fields
      have a trivial `__deinit__` method.

    In practice, it means that the `__deinit__` can be considered as no-op.
    """


# TODO(MOCO-4525): Remove `downcast`
comptime IsTriviallyDeinitable[T: AnyType]: Bool = conforms_to(
    T, TrivialRegisterPassable
) or (conforms_to(T, Deinitable) and downcast[T, Deinitable].__del__is_trivial)
"""Indicates whether `T` is `Deinitable` with a trivial deinitializer.

A deinitializer is trivial when the compiler generates it and all of `T`'s
fields are themselves trivially destructible meaning `__deinit__` is a no-op.
Evaluates to `False` for non-`Deinitable` types.

Parameters:
    T: The type to check.
"""
