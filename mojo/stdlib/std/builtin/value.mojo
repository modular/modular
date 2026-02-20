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
"""Defines core value traits.

These are Mojo built-ins, so you don't need to import them.
"""


@explicit_destroy
trait Movable:
    """The Movable trait denotes a type whose value can be moved.

    Implement the `Movable` trait on `Foo` which requires the `__moveinit__`
    method:

    ```mojo
    struct Foo(Movable):
        fn __init__(out self):
            pass

        fn __moveinit__(out self, deinit take: Self):
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

    fn __moveinit__(out self, deinit take: Self, /):
        """Create a new instance of the value by moving the value of another.

        Args:
            take: The value to move.
        """
        ...

    comptime __moveinit__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of `__moveinit__` is trivial.

    The implementation of `__moveinit__` is considered to be trivial if:
    - The struct has a compiler-generated `__moveinit__` and all its fields
      have a trivial `__moveinit__` method.

    In practice, it means the value can be moved by moving the bits from
    one location to another without side effects.
    """


@explicit_destroy
trait Copyable(Movable):
    """The Copyable trait denotes a type whose value can be explicitly copied.

    Example implementing the `Copyable` trait on `Foo`, which requires the
    `__copyinit__` method:

    ```mojo
    struct Foo(Copyable):
        var s: String

        fn __init__(out self, s: String):
            self.s = s

        fn __copyinit__(out self, copy: Self):
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

    fn __copyinit__(out self, copy: Self, /):
        """Create a new instance of the value by copying an existing one.

        Args:
            copy: The value to copy.
        """
        ...

    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return Self.__copyinit__(self)

    comptime __copyinit__is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of `__copyinit__` is trivial.

    The implementation of `__copyinit__` is considered to be trivial if:
    - The struct has a compiler-generated trivial `__copyinit__` and all its fields
      have a trivial `__copyinit__` method.

    In practice, it means the value can be copied by copying the bits from
    one location to another without side effects.
    """


trait ImplicitlyCopyable(Copyable, ImplicitlyDestructible):
    """A marker trait to permit compiler to insert implicit calls to `__copyinit__`
    in order to make a copy of the object when needed.

    Conforming a type to `ImplicitlyCopyable` gives the Mojo language permission
    to implicitly insert a call to that types copy constructor whenever a borrowed
    instance of the type is passed or assigned where an owned value is required.

    Types that are expensive to copy, or where implicit copying could mask a
    logic error, typically should not be `ImplicitlyCopyable`.

    The `ImplicitlyCopyable` trait is a marker trait, meaning that it does not
    require a type to provide any additional methods or associated aliases to
    conform to this trait. However, all `ImplicitlyCopyable` types are required
    to conform to `Copyable`, which ensures there is only one definition for the
    logic of how a type is copied.

    **Note:** `ImplicitlyCopyable` should only be used to mark structs that may
    be copied implicitly. It should not be used as a trait bound
    (`T: ImplicitlyCopyable`) on functions or types, except in special
    circumstances. Generic code that may perform copies should always use the
    more general `T: Copyable` bound. This ensures that generic code is usable
    with all types that are copyable, regardless of whether they opt-in to
    implicit copying.

    ### Examples

    A type can opt-in to implicit copying by conforming to `ImplicitlyCopyable`
    (in the example below, the compiler also synthesizes a default field-wise
    `__copyinit__()` implementation, as the user didn't provide a definition):

    ```
    @fieldwise_init
    struct Point(ImplicitlyCopyable)
        var x: Int
        var y: Int

    fn main():
        var p = Point(5, 10)

        # Perform an implicit copy of `p
        var p2 = p
    ```
    """

    pass


fn materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile-time parameter into a run-time value.

    Parameters:
        T: The type of the value to materialize.
        value: The compile-time parameter value to materialize.

    Returns:
        The materialized run-time value.
    """
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


trait Defaultable(ImplicitlyDestructible):
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


trait TrivialRegisterPassable(
    ImplicitlyCopyable, ImplicitlyDestructible, Movable, RegisterPassable
):
    """A marker trait to denote the type to be register passable trivial.

     The compiler treats the type that conforms to this trait with the
     following constraints:

     - The type implicitly conforms to Copyable and the compiler synthesizes
     `__copyinit__` that does a memcpy.
     - A trivial `__del__` member is synthesized by the compiler too,
     so the type can’t be a linear type.
     - All declared members are required to also conforms to this trait,
     since you can’t memcpy or trivially destroy a container if one
     of its stored members has a non-trivial copy constructor.
     - You are not allowed to define a custom `__copyinit__` or `__del__`.


     ```mojo
    struct Foo(TrivialRegisterPassable):
        ...
     ```

    """

    pass


trait RegisterPassable(Movable):
    """A marker trait to denote the type to be register passable.

     The compiler treats the type that conforms to this trait with the
     following constraints:

     - the value struct doesn’t have “identity” - you can’t take the
       address of self on read convention methods. This is allows
       the compiler to pass it in registers.

     - The type implicitly conforms to Movable and the compiler synthesizes
       a trivial move constructor. The compiler needs to be able to move around
       values of the type by loading and storing them. A custom defined
       __moveinit__ is not allowed.

     - Compiler checks that any stored member (`var`s) also conforms to this
       trait. It wouldn’t be possible to provide identity for a contained
       member if the container doesn’t have identity.

     - The type can choose whether it wants to be Copyable or not.


     ```mojo
    struct Foo(RegisterPassable):
        ...
     ```

    """

    pass
