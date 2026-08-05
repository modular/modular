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
"""Defines `Movable`, the trait for types whose value can be moved.

This is a Mojo built-in, so you don't need to import it.
"""


@stable(since="1.0")
trait Movable:
    """The Movable trait denotes a type whose value can be moved.

    Implement the `Movable` trait on `Foo` which requires the
    `def __init__(out self, *, deinit move: Self)` method:

    ```mojo
    struct Foo(Movable):
        def __init__(out self):
            pass

        def __init__(out self, *, deinit move: Self):
            print("moving")
    ```

    You can now use the ^ suffix to transfer owned values instead of copying:

    ```mojo
    def return_foo[T: Movable](var foo: T) -> T:
        return foo^

    var foo = Foo()
    var res = return_foo(foo^)
    ```

    ```plaintext
    moving
    ```
    """

    @stable(since="1.0")
    def __init__(out self, *, deinit move: Self):
        """Create a new instance of the value by moving the value of another.

        Args:
            move: The value to move.
        """
        ...

    comptime __move_ctor_is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of move constructor is trivial.

    The implementation of a move constructor is considered to be trivial if:
    - The struct has a compiler-generated trivial move constructor because all
      its fields have trivial move constructors.

    In practice, it means the value can be moved by moving the bits from
    one location to another without side effects.
    """
