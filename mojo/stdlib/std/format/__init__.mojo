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
"""Provides formatting traits for converting types to text.

The `format` package provides traits that control how types format themselves as
text and where that text gets written. The `Writable` trait describes how a
type converts itself to UTF-8 text, while the `Writer` trait accepts formatted
output from writable types. Together, they enable efficient formatting without
unnecessary allocations by writing directly to destinations like files, strings,
or network sockets.

Use this package to implement custom text formatting for your types, control
output representation, or write formatted data directly to various destinations.

## Quick Start

The `Writable` trait has a default implementation that uses reflection to
automatically format all struct fields. Simply declare the trait conformance:

```mojo
@fieldwise_init
struct Point(Writable):
    var x: Float64
    var y: Float64

var p = Point(1.5, 2.7)
print(p)  # Point(x=1.5, y=2.7)
```

## Custom Formatting

Override `write_to()` for custom output:

```mojo
@fieldwise_init
struct Point(Writable):
    var x: Float64
    var y: Float64

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.x, ", ", self.y, ")")

var p = Point(1.5, 2.7)
print(p)  # (1.5, 2.7)
```

## Debug Formatting

Override `write_repr_to()` to provide different debug output:

```mojo
@fieldwise_init
struct Point(Writable):
    var x: Float64
    var y: Float64

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("(", self.x, ", ", self.y, ")")

    fn write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Point(x=", self.x, ", y=", self.y, ")")

var p = Point(1.5, 2.7)
print(p)       # (1.5, 2.7)
print(repr(p)) # Point(x=1.5, y=2.7)
```
"""

from memory import Span
from reflection import struct_field_names, struct_field_count, get_type_name


# ===-----------------------------------------------------------------------===#
# Writer
# ===-----------------------------------------------------------------------===#


trait Writer:
    """A destination for formatted text output.

    `Writer` is implemented by types that can accept UTF-8 formatted text, such
    as strings, files, or network sockets. Types implementing `Writable` write
    their output to a `Writer`.

    The core method is `write_string()`, which accepts a `StringSlice` for
    efficient, allocation-free output. The convenience method `write()` accepts
    multiple `Writable` arguments.

    Example:

    ```mojo
    struct StringBuilder(Writer):
        var s: String

        fn write_string(mut self, string: StringSlice):
            self.s += string

    var builder = StringBuilder("")
    builder.write("Count: ", 42)  # Writes multiple values at once
    print(builder.s)  # Count: 42
    ```
    """

    @deprecated("Writer only supports valid UTF-8, use `write_string` instead")
    @doc_private
    fn write_bytes(mut self, bytes: Span[Byte]):
        self.write_string(StringSlice(unsafe_from_utf8=bytes))

    fn write_string(mut self, string: StringSlice):
        """
        Write a `StringSlice` to this `Writer`.

        Args:
            string: The string slice to write to this Writer.
        """
        ...

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)


# ===-----------------------------------------------------------------------===#
# Writable
# ===-----------------------------------------------------------------------===#


trait Writable:
    """A trait for types that can format themselves as text.

    The `Writable` trait provides a simple, straightforward interface for types
    that need to convert themselves to text. Types implementing `Writable` write
    directly to a `Writer`, making formatting efficient and allocation-free.

    Both `write_to()` and `write_repr_to()` have default implementations that
    use reflection to automatically format all fields. This means simple structs
    can conform to `Writable` without any method implementations:

    ```mojo
    @fieldwise_init
    struct Point(Writable):
        var x: Float64
        var y: Float64

    var p = Point(1.5, 2.7)
    print(p)       # Point(x=1.5, y=2.7)
    print(repr(p)) # Point(x=1.5, y=2.7)
    ```

    Override `write_to()` for custom formatting:

    ```mojo
    @fieldwise_init
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to(self, mut writer: Some[Writer]):
            writer.write("(", self.x, ", ", self.y, ")")

    var p = Point(1.5, 2.7)
    print(p)       # (1.5, 2.7)
    print(repr(p)) # (1.5, 2.7) - uses write_to by default
    ```

    Override both for different normal and debug output:

    ```mojo
    @fieldwise_init
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to(self, mut writer: Some[Writer]):
            writer.write("(", self.x, ", ", self.y, ")")

        fn write_repr_to(self, mut writer: Some[Writer]):
            writer.write("Point(x=", self.x, ", y=", self.y, ")")

    var p = Point(1.5, 2.7)
    print(p)       # (1.5, 2.7)
    print(repr(p)) # Point(x=1.5, y=2.7)
    ```
    """

    fn write_to(self, mut writer: Some[Writer]):
        """Write this value's text representation to a writer.

        This method is called by `print()`, `String()`, and format strings to
        convert the value to text. Override this method to define how your type
        appears when printed or converted to a string.

        The default implementation uses reflection to format all fields as
        `TypeName(field1=value1, field2=value2, ...)`. All fields must conform
        to `Writable`.

        Args:
            writer: The destination for formatted output.

        ## Example

        ```mojo
        fn write_to(self, mut writer: Some[Writer]):
            writer.write("(", self.x, ", ", self.y, ")")
        ```
        """
        # Default implementation using reflection: TypeName(field1=value1, ...)
        comptime names = struct_field_names[Self]()
        writer.write_string(get_type_name[Self]())
        writer.write_string("(")

        @parameter
        for i in range(names.size):

            @parameter
            if i > 0:
                writer.write_string(", ")
            writer.write_string(materialize[names[i]]())
            writer.write_string("=")
            trait_downcast[Writable](__struct_field_ref(i, self)).write_repr_to(
                writer
            )
        writer.write_string(")")

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write this value's debug representation to a writer.

        This method is called by `repr(value)` or the `"{!r}"` format specifier
        and should produce unambiguous, developer-facing output that shows the
        internal state of the value.

        The default implementation calls `write_to()`, providing the same output
        for both normal and debug formatting. Implement this method to provide
        custom debug output.

        Args:
            writer: The destination for formatted output.

        ## Example

        ```mojo
        fn write_repr_to(self, mut writer: Some[Writer]):
            writer.write("Point(x=", self.x, ", y=", self.y, ")")
        ```
        """
        # Default: use normal formatting
        self.write_to(writer)
