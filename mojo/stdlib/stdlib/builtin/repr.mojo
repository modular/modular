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
"""Provide the `repr` function.

The functions and traits provided here are built-ins, so you don't need to import them.
"""

from collections import Set, LinkedList, Deque


trait Representable:
    """A trait that describes a type that has a String representation.

    Any type that conforms to the `Representable` trait can be used with the
    `repr` function. Any conforming type must also implement the `__repr__` method.
    Here is an example:

    ```mojo
    struct Dog(Representable):
        var name: String
        var age: Int

        fn __repr__(self) -> String:
            return "Dog(name=" + repr(self.name) + ", age=" + repr(self.age) + ")"

    var dog = Dog("Rex", 5)
    print(repr(dog))
    # Dog(name='Rex', age=5)
    ```

    The method `__repr__` should compute the "official" string representation of a type.

    If at all possible, this should look like a valid Mojo expression
    that could be used to recreate a struct instance with the same
    value (given an appropriate environment).
    So a returned String of the form `module_name.SomeStruct(arg1=value1, arg2=value2)` is advised.
    If this is not possible, a string of the form `<...some useful description...>`
    should be returned.

    The return value must be a `String` instance.
    This is typically used for debugging, so it is important that the representation is information-rich and unambiguous.

    Note that when computing the string representation of a collection (`Dict`, `List`, `Set`, etc...),
    the `repr` function is called on each element, not the `String()` function.
    """

    fn __repr__(self) -> String:
        """Get the string representation of the type instance, if possible, compatible with Mojo syntax.

        Returns:
            The string representation of the instance.
        """
        pass


fn repr[T: Representable](value: T) -> String:
    """Returns the string representation of the given value.

    Args:
        value: The value to get the string representation of.

    Parameters:
        T: The type of `value`. Must implement the `Representable` trait.

    Returns:
        The string representation of the given value.
    """
    return value.__repr__()


fn repr[T: Representable & Movable & Copyable](value: List[T]) -> String:
    """Returns the string representation of a `List[T]`.

    Args:
        value: A `List` of elements `T`.

    Parameters:
        T: A type that implements `RepresentableCollectionElement`.

    Returns:
        The string representation of `List[T]`.
    """
    # TODO: remove when `List` can conform conditionally to `Representable`.
    return value.__repr__()


fn repr[
    K: KeyElement & Representable,
    V: ExplicitlyCopyable & Movable & Representable,
](value: Dict[K, V]) -> String:
    """Returns the string representation of a `Dict[K,V]`.

    Args:
        value: A `Dict` of keys `K` and elements `V`.

    Parameters:
        K: A type that implements `KeyElement` and `Representable`.
        V: A type that implements `Movable`, `Copyable` and `Representable`.

    Returns:
        The string representation of `Dict[K,V]`.
    """
    # TODO: remove when `Dict` can conform conditionally to `Representable`.
    return value.__str__()


fn repr[U: KeyElement & Representable](value: Set[U]) -> String:
    """Returns the string representation of an `Set[U]`.

    Args:
        value: A `Set` of elements `U`.

    Parameters:
        U: A type that implements `KeyElement` and `Representable`.

    Returns:
        The string representation of `Set[U]`.
    """
    # TODO: remove when `Set` can conform conditionally to `Representable`.
    return value.__repr__()


fn repr[U: Representable & Copyable & Movable](value: Optional[U]) -> String:
    """Returns the string representation of an `Optional[U]`.

    Args:
        value: A `Optional` of element type `U`.

    Parameters:
        U: A type that implements `Movable`, `Copyable` and `Representable`.

    Returns:
        The string representation of `Optional[U]`.
    """
    # TODO: remove when `Optional` can conform conditionally to `Representable`.
    return value.__repr__()


fn repr[
    U: ExplicitlyCopyable & Movable & Writable
](value: LinkedList[U]) -> String:
    """Returns the string representation of an `LinkedList[U]`.

    Args:
        value: A `LinkedList` of element type `U`.

    Parameters:
        U: A type that implements `Movable`, `ExplicitlyCopyable` and `Writable`.

    Returns:
        The string representation of `LinkedList[U]`.
    """
    # TODO: remove when `LinkedList` can conform conditionally to `Representable`.
    return value.__repr__()


fn repr[
    T: Representable & ExplicitlyCopyable & Movable
](value: Deque[T]) -> String:
    """Returns the string representation of an `Deque[U]`.

    Args:
        value: A `Deque` of element type `U`.

    Parameters:
        T: A type that implements `Movable`, `ExplicitlyCopyable` and `Representable`.

    Returns:
        The string representation of `Deque[U]`.
    """
    # TODO: remove when `Deque` can conform conditionally to `Representable`.
    return value.__repr__()


fn repr(value: None) -> String:
    """Returns the string representation of `None`.

    Args:
        value: A `None` value.

    Returns:
        The string representation of `None`.
    """
    return "None"
