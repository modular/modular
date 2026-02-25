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
"""Implements utilities to capture and represent source code location.

This module provides compile-time and runtime introspection of source locations:

- `SourceLocation` - A struct holding file name, line, and column information.
- `source_location()` - Returns the location where this function is called.
- `call_location()` - Returns the caller's location (for use in inlined functions).

These utilities are useful for error reporting, logging, debugging, and building
custom assertion functions that report meaningful locations to users.

Example using `source_location()` to get the current location:

```mojo
from reflection import source_location

fn main():
    var loc = source_location()
    print(loc)  # Prints: /path/to/file.mojo:5:15
    print("Line:", loc.line, "Column:", loc.col)
```

Example using `call_location()` for a custom assertion that reports the
caller's location. Note that `@always_inline` is required for `call_location()`
to work - the function must be inlined so the compiler can capture the caller's
location:

```mojo
from reflection import call_location

@always_inline  # Required for call_location() to work
fn my_assert(cond: Bool, msg: String = "assertion failed") raises:
    if not cond:
        raise Error(call_location().prefix(msg))

def main():
    var x = 5
    my_assert(x > 10, "x must be > 10")  # Error points to THIS line
```
"""


@fieldwise_init
struct SourceLocation(TrivialRegisterPassable, Writable):
    """Type to carry file name, line, and column information.

    This struct stores source location data and provides utilities for formatting
    location-prefixed messages, which is useful for error reporting and debugging.

    Example:

    ```mojo
    from reflection import source_location, SourceLocation

    fn main():
        # Get current location
        var loc = source_location()
        print(loc)  # Prints: /path/to/file.mojo:6:19

        # Use prefix() for error-style messages
        print(loc.prefix("something went wrong"))
        # Prints: At /path/to/file.mojo:6:19: something went wrong

        # Access individual fields
        print("File:", loc.file_name)
        print("Line:", loc.line)
        print("Column:", loc.col)
    ```
    """

    var line: Int
    """The line number (1-indexed)."""
    var col: Int
    """The column number (1-indexed)."""
    var file_name: StaticString
    """The file name."""

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the source location.

        Returns:
            A string in the format "file_name:line:col".
        """
        return String.write(self)

    @no_inline
    fn prefix[T: Writable](self, msg: T) -> String:
        """Returns the given message prefixed with the source location.

        Parameters:
            T: The type of the message.

        Args:
            msg: The message to attach the prefix to.

        Returns:
            A string in the format "At file:line:col: msg".
        """
        return String("At ", self, ": ", msg)

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats the source location to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(self.file_name, ":", self.line, ":", self.col)


@always_inline("nodebug")
fn source_location() -> SourceLocation:
    """Returns the location for where this function is called.

    This currently doesn't work when called in a parameter expression.

    Returns:
        The location information of the `source_location()` call.

    Example:

    ```mojo
    from reflection import source_location

    fn log_message(msg: String):
        var loc = source_location()
        print("[", loc.file_name, ":", loc.line, "]", msg)

    fn main():
        log_message("hello")  # Prints: [ /path/to/file.mojo : 4 ] hello
    ```
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount = Int(0)._mlir_value,
        _type = Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()

    return SourceLocation(
        Int(mlir_value=line),
        Int(mlir_value=col),
        StaticString(file_name),
    )


@always_inline("nodebug")
fn call_location[*, inline_count: Int = 1]() -> SourceLocation:
    """Returns the location for where the caller of this function is called.

    An optional `inline_count` parameter can be specified to skip over that many
    levels of calling functions.

    This should only be used when enclosed in a series of `@always_inline` or
    `@always_inline("nodebug")` function calls, where the layers of calling
    functions is no fewer than `inline_count`.

    For example, when `inline_count = 1`, only the caller of this function needs
    to be `@always_inline` or `@always_inline("nodebug")`. This function will
    return the source location of the caller's invocation.

    When `inline_count = 2`, the caller of the caller of this function also
    needs to be inlined. This function will return the source location of the
    caller's caller's invocation.

    This currently doesn't work when the `inline_count`-th wrapping caller is
    called in a parameter expression.

    Parameters:
        inline_count: The number of inline call levels to skip.

    Returns:
        The location information of where the caller of this function (i.e. the
        function whose body `call_location()` is used in) is called.

    Example:

    ```mojo
    from reflection import call_location

    @always_inline  # Required for call_location() to work
    fn assert_positive(value: Int) raises:
        # call_location() returns where assert_positive() was called,
        # not where call_location() itself is called.
        if value <= 0:
            raise Error(call_location().prefix("value must be positive"))

    fn main():
        try:
            assert_positive(-1)  # Error will point to THIS line
        except e:
            print(e)
    ```
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount = inline_count._mlir_value,
        _type = Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()

    return SourceLocation(
        Int(mlir_value=line),
        Int(mlir_value=col),
        StaticString(file_name),
    )
