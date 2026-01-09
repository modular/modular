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
"""Text formatting: traits for writing types as human-readable UTF-8 strings.

The `format` package provides traits that control how types format themselves as
text and where that text gets written. The `Writable` trait describes how a
type converts itself to UTF-8 text, while the `Writer` trait accepts formatted
output from writable types. Together, they enable efficient formatting without
unnecessary allocations by directly to destinations like files, strings, or
network sockets.

Use this package to implement custom text formatting for your types, control
output representation, or write formatted data directly to various destinations.
"""

from memory import Span


trait Writer:
    """Describes a type that can be written to by any type that implements the
    `write_to` function.

    This enables you to write one implementation that can be written to a
    variety of types such as file descriptors, strings, network locations etc.
    The types are written as a `StringSlice`, so the `Writer` can avoid
    allocations depending on the requirements. There is also a general `write`
    that takes multiple args that implement `write_to`.

    Example:

    ```mojo
    @fieldwise_init
    struct NewString(Writer, Writable, ImplicitlyCopyable):
        var s: String

        # Writer requirement to write a String
        fn write_string(mut self, string: StringSlice):
            self.s += string

        # Also make it Writable to allow `print` to write the inner String
        fn write_to(self, mut writer: Some[Writer]):
            writer.write(self.s)


    @fieldwise_init
    struct Point(Writable, ImplicitlyCopyable):
        var x: Int
        var y: Int

        # Pass multiple args to the Writer. The Int and StaticString types
        # call `writer.write_string` in their own `write_to` implementations.
        fn write_to(self, mut writer: Some[Writer]):
            writer.write("Point(", self.x, ", ", self.y, ")")


    fn main():
        var point = Point(1, 2)
        var new_string = NewString(String(point))
        new_string.write("\\n", Point(3, 4))
        print(new_string)
    ```

    Output:

    ```plaintext
    Point(1, 2)
    Point(3, 4)
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
    """The `Writable` trait describes how a type is written into a `Writer`.

    The `Writable` trait is designed for efficient output operations. It
    differs from [`Stringable`](/mojo/std/builtin/str/Stringable) in that
    `Stringable` merely converts a type to a `String` type, whereas `Writable`
    directly writes the type to an output stream, making it more efficient for
    output operations like [`print()`](/mojo/std/io/io/print).

    To make your type conform to `Writable`, you must implement `write_to()`
    which takes `self` and a type conforming to
    [`Writer`](/mojo/std/io/write/Writer):

    ```mojo
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to(self, mut writer: Some[Writer]):
            var string = "Point"
            # Write a single `StringSlice`:
            writer.write_string(string)
            # Pass multiple args that implement `Writable`:
            writer.write("(", self.x, ", ", self.y, ")")
    ```
    """

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats the string representation of this type to the provided Writer.

        For example, when you pass your type to
        [`print()`](/mojo/std/io/io/print/), it calls `write_to()` on your
        type and the `writer` is the
        [`FileDescriptor`](/mojo/std/io/file_descriptor/FileDescriptor/)
        that `print()` is writing to.

        Args:
            writer: The type conforming to `Writable`.
        """
        ...
