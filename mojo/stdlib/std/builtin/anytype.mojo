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
"""Defines the core traits for object lifetime management in Mojo.

This module provides the foundational traits that define how objects are created,
managed and destroyed in Mojo:

- `AnyType`: The most basic trait that all types extend by default.
   Types with this trait have no destructor and no lifetime management.

- `ImplicitlyDestructible`: The base trait for types that require lifetime
   management through destructors. Any type that needs cleanup when it goes out
   of scope should implement this trait.

These traits are built into Mojo and do not need to be imported.
"""

# ===----------------------------------------------------------------------=== #
#  AnyType
# ===----------------------------------------------------------------------=== #


trait AnyType:
    """The most basic trait that all Mojo types extend by default.

    All Mojo struct types always conform to `AnyType`. This trait imposes no
    requirements on the types that conform to it, not even that they provide
    a `__del__()` implicit destructor.

    A type that conforms to `AnyType` but not to `ImplicitlyDestructible` is
    called a linear type, also known as a non-implicitly-destructible type.

    Generic code will commonly want to use `T: ImplicitlyDestructible` instead
    of `T: AnyType`.

    # `AnyType`, Object Destructors, and Linear Types

    Mojo's `AnyType` is a lower-level, more powerful building block than is
    found in many mainstream programming languages today.

    In most programming languages that enforce strong object initialization and
    destruction lifecycle semantics ("RAII"), the programmer is required to
    define a destructor function that will "tear down" an object instance and
    release any resources it has logical ownership over. In such languages, the
    compiler is permitted to destroy an object instance *implicitly* whenever it
    determines that the instance is no longer used, or has "gone out of scope".

    Another way to state the above is that, in many programming languages, the
    minimum requirement of all types is that they provide *at least* a
    trivial (possibly empty) destructor function. Mojo's `AnyType` is more
    basic than that seemingly minimum requirement.

    *Unlike* in programming langauges the reader is likely to be familiar with,
    Mojo enforces strong object lifecycles, but does *not* require that a type
    provide an implicitly-callable destructor function. Instead, a type may
    choose to provide only named, explicitly-callable destructor methods.

    Said another way, Mojo gives type authors a type to provide either:

    * A `__del__()` destructor method that the compiler may call implicitly
      whenever an owned object instances has no further uses. Such types
      conform to `ImplicitlyDestructible`.

    * Named destructor methods that type user must choose to call explicitly.
      Failing to explicitly destroy such a type will lead to a compile-time
      error, requiring the programmer to chose how to destroy the object or
      keep it alive for longer.

    (Technically, a type can choose to provide *neither* implicit nor named
    destructors, but an instance of such a type would effectively be a
    "hot potato", getting tossed along forever without any way to "consume" the
    instance.)

    Named destructors give library type authors a powerful tool to enforce
    correctness and safety invariants. A type that provides only named
    destructor methods (a linear type) makes object destruction the explicit
    choice of the downstream user, instead of something done implicitly when the
    compiler thinks it is appropriate.

    Linear types can act as a guard that some explicit action must be performed
    sometime "in the future" after initial object construction.

    The following is a simple example of a non-implicilty-destructible type with
    a named destructor method:

    ```mojo
    @explicit_destroy
    struct FileBuffer:
        fn __init__(out self, path: Path):
            # ... open the file at the specified `path` ...

        fn write(self, data: Some[Writable]):
            # ... buffered write of the specified data to this file ...

        fn save_and_close(deinit self):
            # ... save out the buffered data ...

    # 🔴 ERROR: 'file' abandoned without being explicitly destroyed
    fn write_greeting_to_file(var file: FileBuffer):
        file.write("Hello there!")

        # 🟢 FIX: add `file^.save_and_close()`
    ```

    In the above example, the user is saved from forgetting to flush any
    buffered data because `FileBuffer` cannot simply be "dropped on the floor" —
    the programmer must choose to call `FileBuffer.save_and_close()` when they
    are finished with `file`.

    The `FileBuffer.save_and_close()` method is special because it takes
    `deinit self`. The `deinit` argument convention is special, and signals that
    the object is consumed by calling that method, with no further tear down
    logic required.
    """

    pass


trait ImplicitlyDestructible:
    """A trait for types that require lifetime management through destructors.

    The `ImplicitlyDestructible` trait is fundamental to Mojo's memory
    management system. It indicates that a type has a destructor that needs to
    be called when instances go out of scope. This is essential for types that
    own resources like memory, file handles, or other system resources that need
    proper cleanup.

    By default, all Mojo types implement `ImplicitlyDestructible`, unless they
    opt-in to explicit named destructor methods using `@explicit_destroy`.

    Key aspects:

    - Any type with an implicit `__del__()` destructor must implement this trait
    - The destructor (`__del__`) is called automatically when an instance's
      lifetime ends
    - Composition of types with implicit destructors automatically get an
      implicit destructor

    Example:

    ```mojo
    struct ResourceOwner(ImplicitlyDestructible):
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

    comptime __del__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation of `__del__` is trivial.

    The implementation of `__del__` is considered to be trivial if:
    - The struct has a compiler-generated trivial destructor and all its fields
      have a trivial `__del__` method.

    In practice, it means that the `__del__` can be considered as no-op.
    """


comptime __SomeImpl[Trait: __TypeOfAllTypes, T: Trait] = T

comptime Some[Trait: __TypeOfAllTypes] = __SomeImpl[Trait]
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

Parameters:
    Trait: The trait or trait composition that the argument type must implement.
"""
