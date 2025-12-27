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
"""Implements the ReadOnly datatype."""


struct ReadOnly[T: Movable & ImplicitlyDestructible]:
    """A wrapper type to provide a runtime read-only value.

    `ReadOnly` wraps a value that is initialized at runtime and can't be
    modified thereafter. It provides an immutable reference to the underlying
    value.

    Example:
    ```mojo
    from time import current_time_ns
    from utils import ReadOnly

    def main():
        ref start_time = ReadOnly(current_time_ns())[]
        # value is known at runtime but must stay immutable thereafter
        # ... any code here is guaranteed to not modify start_time ...
        # start_time = 0  # compile-time error
        var duration = current_time_ns() - start_time
        print("Duration (ns): ", duration)
    ```

    Prefer `comptime` for constant values that are known at compilation time:

    ```mojo
    def main():
        comptime magic_number = 42
        # value is known at compilation time
        # ... any code here is guaranteed to not modify magic_number ...
        print("Magic Number: ", magic_number)
    ```
    If the value is provided through a default (`read`) argument convention
    `RuntimeConst` might not be needed either.

    Parameters:
        T: The type of the value being wrapped. Must be `Movable` and
           `ImplicitlyDestructible`.
    """

    var _value: Self.T
    """The wrapped value."""

    @always_inline
    fn __init__(out self, var value: Self.T):
        """Initializes a `ReadOnly` instance with the provided value.

        Args:
            value: The value to be wrapped as read-only.

        Returns:
            A new `ReadOnly` instance wrapping the provided value.
        """

        self._value = value^

    @always_inline
    fn __getitem__(self) -> ref [self._value] Self.T:
        """Get the underlying value as an immutable reference.

        Returns:
            An immutable reference to the underlying value.
        """
        return self._value
