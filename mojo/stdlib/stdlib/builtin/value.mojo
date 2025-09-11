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
"""Defines core value traits.

These are Mojo built-ins, so you don't need to import them.
"""


trait Movable:
    """The Movable trait denotes a type whose value can be moved.

    Implement the `Movable` trait on `Foo` which requires the `__moveinit__`
    method:

    ```mojo
    struct Foo(Movable):
        fn __init__(out self):
            pass

        fn __moveinit__(out self, deinit existing: Self):
            print("moving")
    ```

    You can now use the ^ suffix to move the object instead of copying
    it inside generic functions:

    ```mojo
    fn return_foo[T: Movable](var foo: T) -> T:
        return foo^

    var foo = Foo()
    var res = return_foo(foo^)
    ```

    ```plaintext
    moving
    ```
    """

    fn __moveinit__(out self, deinit existing: Self, /):
        """Create a new instance of the value by moving the value of another.

        Args:
            existing: The value to move.
        """
        ...

    alias __moveinit__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of `__moveinit__` is trivial.

    The implementation of `__moveinit__` is considered to be trivial if:
    - The struct has a compiler-generated `__moveinit__` and all its fields
      have a trivial `__moveinit__` method.

    In practice, it means the value can be moved by moving the bits from
    one location to another without side effects.
    """


trait Copyable:
    """The Copyable trait denotes a type whose value can be explicitly copied.

    Example implementing the `Copyable` trait on `Foo`, which requires the
    `__copyinit__` method:

    ```mojo
    struct Foo(Copyable):
        var s: String

        fn __init__(out self, s: String):
            self.s = s

        fn __copyinit__(out self, other: Self):
            print("copying value")
            self.s = other.s
    ```

    You can now copy objects inside a generic function:

    ```mojo
    fn copy_return[T: Copyable](foo: T) -> T:
        var copy = foo.copy()
        return copy^

    var foo = Foo("test")
    var res = copy_return(foo)
    ```

    ```plaintext
    copying value
    ```
    """

    fn __copyinit__(out self, existing: Self, /):
        """Create a new instance of the value by copying an existing one.

        Args:
            existing: The value to copy.
        """
        ...

    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return Self.__copyinit__(self)

    alias __copyinit__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of `__copyinit__` is trivial.

    The implementation of `__copyinit__` is considered to be trivial if:
    - The struct has a compiler-generated trivial `__copyinit__` and all its fields
      have a trivial `__copyinit__` method.

    In practice, it means the value can be copied by copying the bits from
    one location to another without side effects.
    """


trait ImplicitlyCopyable(Copyable):
    """A marker trait to permit compiler to insert implicit calls to `__copyinit__`
    in order to make a copy of the object when needed.
    """

    pass


@deprecated(
    "Use `Copyable` or `ImplicitlyCopyable` instead. `Copyable` on its own no"
    " longer implies implicit copyability."
)
alias ExplicitlyCopyable = Copyable


fn materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile time parameter into a runtime value."""
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


trait Defaultable:
    """The `Defaultable` trait describes a type with a default constructor.

    Implementing the `Defaultable` trait requires the type to define
    an `__init__` method with no arguments:

    ```mojo
    struct Foo(Defaultable):
        var s: String

        fn __init__(out self):
            self.s = "default"
    ```

    You can now construct a generic `Defaultable` type:

    ```mojo
    fn default_init[T: Defaultable]() -> T:
        return T()

    var foo = default_init[Foo]()
    print(foo.s)
    ```

    ```plaintext
    default
    ```
    """

    fn __init__(out self):
        """Create a default instance of the value."""
        ...
