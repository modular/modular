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
"""Provides the `reversed` function for reverse iteration over collections.

These are Mojo built-ins, so you don't need to import them.
"""

from std.collections import Deque
from std.collections.deque import _DequeIter
from std.collections.dict import _DictEntryIter, _DictKeyIter, _DictValueIter
from std.collections.array import _ArrayIter
from std.hash import Hasher

from std.collections.span import Span, _SpanIter

# ===----------------------------------------------------------------------=== #
#  Reversible
# ===----------------------------------------------------------------------=== #


trait ReversibleRange:
    """
    The `ReversibleRange` trait describes a range that can be reversed.

    Any type that conforms to `ReversibleRange` works with the builtin
    [`reversed()`](/docs/std/builtin/reversed/reversed/) functions.

    The `ReversibleRange` trait requires the type to define the `__reversed__()`
    method and a `ReversedType` iterator, so each conforming range can return
    its own reversed iterator type instead of a single hard-coded one.

    **Note**: iterators are currently non-raising.
    """

    # TODO: general `Reversible` trait that returns an iterator.
    # iterators currently check __len__() instead of raising an exception
    # so there is no ReversibleRaising trait yet.

    comptime ReversedType: Iterator
    """The iterator type returned by `__reversed__()`."""

    def __reversed__(self) -> Self.ReversedType:
        """Returns a reverse iterator for the conforming type.

        Returns:
            An iterator over the type's elements in reverse order.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  reversed
# ===----------------------------------------------------------------------=== #


def reversed[T: ReversibleRange](value: T) -> T.ReversedType:
    """Returns an iterator over the elements in reverse order.

    The iterator comes from the value's `__reversed__()` method, so any type
    that conforms to `ReversibleRange` works here. Iterators are currently
    non-raising.

    Parameters:
        T: The type of the value. Must conform to `ReversibleRange`.

    Args:
        value: The range to iterate in reverse.

    Returns:
        The reverse iterator of the range.
    """
    return value.__reversed__()


def reversed[T: Copyable](ref value: List[T]) -> type_of(value.__reversed__()):
    """Returns an iterator over the list's elements in reverse order.

    Parameters:
        T: The element type of the list.

    Args:
        value: The list to iterate in reverse.

    Returns:
        An iterator over the list's elements in reverse order.

    For example:

    ```mojo
    var numbers = [1, 2, 3]
    for number in reversed(numbers):
        print(number)  # 3, 2, 1
    ```
    """
    return value.__reversed__()


def reversed[
    T: Copyable, size: Int
](ref value: Array[T, size]) -> _ArrayIter[T, size, origin_of(value), False]:
    """Returns an iterator over the array's elements in reverse order.

    Parameters:
        T: The element type of the array.
        size: The number of elements in the array.

    Args:
        value: The array to iterate in reverse.

    Returns:
        An iterator over the array's elements in reverse order.
    """
    return value.__reversed__()


def reversed[
    T: Copyable & Deinitable
](ref value: Deque[T]) -> _DequeIter[T, origin_of(value), False]:
    """Returns an iterator over the deque's elements in reverse order.

    Parameters:
        T: The element type of the deque.

    Args:
        value: The deque to iterate in reverse.

    Returns:
        An iterator over the deque's elements in reverse order.
    """
    return value.__reversed__()


def reversed[
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher & Defaultable & Deinitable,
](ref value: Dict[K, V, H],) -> _DictKeyIter[K, V, H, origin_of(value), False]:
    """Returns an iterator over the dict's keys in reverse order.

    Parameters:
        K: The key type of the dict.
        V: The value type of the dict.
        H: The hasher type of the dict.

    Args:
        value: The dict to iterate in reverse.

    Returns:
        An iterator over the dict's keys in reverse order.

    For example:

    ```mojo
    var scores = {"ana": 90, "ben": 85}
    for name in reversed(scores):
        print(name)  # ben, ana
    ```
    """
    return value.__reversed__()


def reversed[
    dict_mutability: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher & Defaultable & Deinitable,
    dict_origin: Origin[mut=dict_mutability],
](ref value: _DictValueIter[K, V, H, dict_origin]) -> _DictValueIter[
    K, V, H, dict_origin, False
]:
    """Returns an iterator over the dict values in reverse order.

    Use it on a `values()` view to walk the values back to front.

    Parameters:
        dict_mutability: Whether the reference to the dict values is mutable.
            Inferred; it can't be set at the call site.
        K: The key type of the dict.
        V: The value type of the dict.
        H: The hasher type of the dict.
        dict_origin: The origin of the dict values. Inferred.

    Args:
        value: The dict values to iterate in reverse.

    Returns:
        An iterator over the dict values in reverse order.
    """
    return value.__reversed__()


def reversed[
    dict_mutability: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher & Defaultable & Deinitable,
    dict_origin: Origin[mut=dict_mutability],
](ref value: _DictEntryIter[K, V, H, dict_origin]) -> _DictEntryIter[
    K, V, H, dict_origin, False
]:
    """Returns an iterator over the dict entries in reverse order.

    Use it on an `items()` view to walk the key-value entries back to front.

    Parameters:
        dict_mutability: Whether the reference to the dict items is mutable.
            Inferred; it can't be set at the call site.
        K: The key type of the dict.
        V: The value type of the dict.
        H: The hasher type of the dict.
        dict_origin: The origin of the dict items. Inferred.

    Args:
        value: The dict items to iterate in reverse.

    Returns:
        An iterator over the dict items in reverse order.
    """
    var src = value.src
    return _DictEntryIter[K, V, H, dict_origin, False](
        len(src[]._order) - 1, 0, src
    )


@inline(.always)
def reversed[
    T: Copyable
](value: Span[T, _]) -> _SpanIter[T, value.origin, forward=False]:
    """Returns an iterator over the span's elements in reverse order.

    Parameters:
        T: The element type of the span.

    Args:
        value: The span to iterate in reverse.

    Returns:
        An iterator over the span's elements in reverse order.
    """
    return rebind[_SpanIter[T, value.origin, forward=False]](
        value.__reversed__()
    )
