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
"""Defines the core traits for object lifetime management in Mojo.

This module provides the foundational traits that define how objects are created,
managed and destroyed in Mojo:

- `UnknownDestructibility`: The most basic trait that all types extend by default.
   Types with this trait have no destructor and no lifetime management.

- `AnyType`: The base trait for types that require lifetime management through
   destructors. Any type that needs cleanup when it goes out of scope should
   implement this trait.

- `ImplicitlyDestructible`: An alias for `AnyType` to help with the transition
   to linear types. Use this when you want to be explicit about a type having
   a destructor.

These traits are built into Mojo and do not need to be imported.
"""

# ===----------------------------------------------------------------------=== #
#  AnyType
# ===----------------------------------------------------------------------=== #


# TODO(MOCO-1468): Add @explicit_destroy here so we get an error message,
#     preferably one that mentions a link the user can go to to learn about
#     linear types.
trait UnknownDestructibility:
    """The most basic trait that all Mojo types extend by default.

    This trait indicates that a type has no destructor and therefore no lifetime
    management. It is the default for all types unless they explicitly implement
    `AnyType` or `ImplicitlyDestructible`.

    Types with this trait:
    - Have no `__del__` method
    - Do not perform any cleanup when they go out of scope
    - Are suitable for simple value types that don't own resources

    For types that need cleanup when they are destroyed, use `ImplicitlyDestructible`
    or `AnyType` instead.
    """

    pass


trait AnyType:
    """A trait for types that require lifetime management through destructors.

    The `AnyType` trait is fundamental to Mojo's memory management system. It indicates
    that a type has a destructor that needs to be called when instances go out of scope.
    This is essential for types that own resources like memory, file handles, or other
    system resources that need proper cleanup.

    Key aspects:

    - Any type with a destructor must implement this trait
    - The destructor (`__del__`) is called automatically when an instance's lifetime ends
    - Composition of types with destructors automatically gets a destructor
    - All Mojo structs and traits inherit from `AnyType` by default unless they specify
        `@explicit_destroy`

    Example:

    ```mojo
    struct ResourceOwner(AnyType):
        var ptr: UnsafePointer[Int]

        fn __init__(out self, size: Int):
            self.ptr = UnsafePointer[Int].alloc(size)

        fn __del__(deinit self):
            # Clean up owned resources
            self.ptr.free()
    ```

    Best practices:

    - Implement this trait when your type owns resources that need cleanup
    - Ensure the destructor properly frees all owned resources
    - Consider using `@explicit_destroy` for types that should never have destructors
    - Use composition to automatically handle nested resource cleanup
    """

    fn __del__(deinit self, /):
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

    alias __del__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation of `__del__` is trivial.

    The implementation of `__del__` is considered to be trivial if:
    - The struct has a compiler-generated trivial destructor and all its fields
      have a trivial `__del__` method.

    In practice, it means that the `__del__` can be considered as no-op.
    """


# A temporary alias to help with the linear types transition, see
# https://www.notion.so/modularai/Linear-Types-14a1044d37bb809ab074c990fe1a84e3.
alias ImplicitlyDestructible = AnyType


alias __SomeImpl[Trait: AnyTrivialRegType, T: Trait] = T

alias Some[Trait: AnyTrivialRegType] = __SomeImpl[Trait]
"""An alias allowing users to tersely express that a function argument is an
instance of a type that implements a trait or trait composition.

For example, instead of writing

```mojo
fn foo[T: Intable, //](x: T) -> Int:
    return x.__int__()
```

one can write:

```mojo
fn foo(x: Some[Intable]) -> Int:
    return x.__int__()
```
"""
