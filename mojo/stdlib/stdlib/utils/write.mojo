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
"""Establishes the contract between `Writer` and `Writable` types."""

from bit import byte_swap
from builtin.io import _printf
from collections import InlineArray
from sys.info import is_amd_gpu, is_gpu, is_nvidia_gpu
from memory import Span, UnsafePointer, memcpy, bitcast
from os import abort
from collections import InlineArray
from sys.info import is_amd_gpu, is_gpu, is_nvidia_gpu
from sys import alignof
from memory import Span, UnsafePointer, memcpy
from sys.param_env import env_get_int
from collections.string.string_slice import _get_kgen_string


# ===-----------------------------------------------------------------------===#

alias HEAP_BUFFER_BYTES = env_get_int["HEAP_BUFFER_BYTES", 2048]()
"""How much memory to pre-allocate for the heap buffer, will abort if exceeded."""
alias STACK_BUFFER_BYTES = env_get_int["STACK_BUFFER_BYTES", 4096]()
"""The size of the stack buffer for IO operations from CPU."""


trait Writer:
    """Describes a type that can be written to by any type that implements the
    `write_to` function.

    This enables you to write one implementation that can be written to a
    variety of types such as file descriptors, strings, network locations etc.
    The types are written as a `Span[Byte]`, so the `Writer` can avoid
    allocations depending on the requirements. There is also a general `write`
    that takes multiple args that implement `write_to`.

    Example:

    ```mojo
    from memory import Span

    @fieldwise_init
    struct NewString(Writer, Writable, Copyable, Movable):
        var s: String

        # Writer requirement to write a Span of Bytes
        fn write_bytes(mut self, bytes: Span[Byte, _]):
            self.s._iadd(bytes)

        # Writer requirement to take multiple args
        fn write[*Ts: Writable](mut self, *args: *Ts):
            @parameter
            for i in range(args.__len__()):
                args[i].write_to(self)

        # Also make it Writable to allow `print` to write the inner String
        fn write_to[W: Writer](self, mut writer: W):
            writer.write(self.s)


    @fieldwise_init
    struct Point(Writable, Copyable, Movable):
        var x: Int
        var y: Int

        # Pass multiple args to the Writer. The Int and StaticString types
        # call `writer.write_bytes` in their own `write_to` implementations.
        fn write_to[W: Writer](self, mut writer: W):
            writer.write("Point(", self.x, ", ", self.y, ")")

        # Enable conversion to a String using `String(point)`
        fn __str__(self) -> String:
            return String.write(self)


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

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a `Span[Byte]` to this `Writer`.

        Args:
            bytes: The string slice to write to this Writer. Must NOT be
              null-terminated.
        """
        ...

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """
        ...
        # TODO: When have default implementations on traits, we can use this:
        # @parameter
        # for i in range(args.__len__()):
        #     args[i].write_to(self)
        #
        # To only have to implement `write_bytes` to make a type a valid Writer


# ===-----------------------------------------------------------------------===#
# Writable
# ===-----------------------------------------------------------------------===#


trait Writable:
    """The `Writable` trait describes how a type is written into a `Writer`.

    You must implement `write_to` which takes `self` and a type conforming to
    `Writer`:

    ```mojo
    struct Point(Writable):
        var x: Float64
        var y: Float64

        fn write_to[W: Writer](self, mut writer: W):
            var string = "Point"
            # Write a single `Span[Byte]`:
            writer.write_bytes(string.as_bytes())
            # Pass multiple args that can be converted to a `Span[Byte]`:
            writer.write("(", self.x, ", ", self.y, ")")
    ```
    """

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats the string representation of this type to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The type conforming to `Writable`.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


struct _WriteBufferHeap(Writer, Writable):
    var data: UnsafePointer[Byte]
    var pos: Int

    fn __init__(out self):
        alias alignment: Int = alignof[Byte]() if is_gpu() else 1
        self.data = __mlir_op.`pop.stack_allocation`[
            count = HEAP_BUFFER_BYTES.value,
            _type = UnsafePointer[Byte]._mlir_type,
            alignment = alignment.value,
        ]()
        self.pos = 0

    fn write_list[
        T: Copyable & Movable & Writable, //
    ](mut self, values: List[T, *_], *, sep: StaticString = StaticString()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    @always_inline
    fn write_bytes(mut self, bytes: Span[UInt8, _]):
        len_bytes = len(bytes)
        if len_bytes + self.pos > HEAP_BUFFER_BYTES:
            _printf[
                "HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D"
                " HEAP_BUFFER_BYTES=4096`\n"
            ]()
            abort()
        memcpy(self.data + self.pos, bytes.unsafe_ptr(), len_bytes)
        self.pos += len_bytes

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write_bytes(
            Span[Byte, __origin_of(self)](ptr=self.data, length=self.pos)
        )


struct _WriteBufferStack[origin: MutableOrigin, W: Writer, //](Writer):
    var data: InlineArray[UInt8, STACK_BUFFER_BYTES]
    var pos: Int
    var writer: Pointer[W, origin]

    @implicit
    fn __init__(out self, ref [origin]writer: W):
        self.data = InlineArray[UInt8, STACK_BUFFER_BYTES](uninitialized=True)
        self.pos = 0
        self.writer = Pointer(to=writer)

    fn write_list[
        T: Copyable & Movable & Writable, //
    ](mut self, values: List[T, *_], *, sep: String = String()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    fn flush(mut self):
        self.writer[].write_bytes(
            Span(ptr=self.data.unsafe_ptr(), length=self.pos)
        )
        self.pos = 0

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        len_bytes = len(bytes)
        # If span is too large to fit in buffer, write directly and return
        if len_bytes > STACK_BUFFER_BYTES:
            self.flush()
            self.writer[].write_bytes(bytes)
            return
        # If buffer would overflow, flush writer and reset pos to 0.
        elif self.pos + len_bytes > STACK_BUFFER_BYTES:
            self.flush()
        # Continue writing to buffer
        memcpy(self.data.unsafe_ptr() + self.pos, bytes.unsafe_ptr(), len_bytes)
        self.pos += len_bytes

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


# fmt: off
alias _hex_table = SIMD[DType.uint8, 16](
    ord("0"), ord("1"), ord("2"), ord("3"), ord("4"),
    ord("5"), ord("6"), ord("7"), ord("8"), ord("9"),
    ord("a"), ord("b"), ord("c"), ord("d"), ord("e"), ord("f"),
)
# fmt: on


@always_inline
fn _hex_digits_to_hex_chars(ptr: UnsafePointer[Byte], decimal: Scalar):
    """Write a fixed width hexadecimal value into an uninitialized pointer
    location, assumed to be large enough for the value to be written.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from utils.write import _hex_digits_to_hex_chars
    items = List[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0)
    alias S = StringSlice[__origin_of(items)]
    ptr = items.unsafe_ptr()
    _hex_digits_to_hex_chars(ptr, UInt32(ord("🔥")))
    assert_equal("0001f525", S(ptr=ptr, length=8))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt16(ord("你")))
    assert_equal("4f60", S(ptr=ptr, length=4))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt8(ord("Ö")))
    assert_equal("d6", S(ptr=ptr, length=2))
    ```
    .
    """

    alias size = decimal.dtype.sizeof()
    var data: SIMD[DType.uint8, size]

    @parameter
    if size == 1:
        data = bitcast[DType.uint8, size](decimal)
    else:
        data = bitcast[DType.uint8, size](byte_swap(decimal))
    var nibbles = (data >> 4).interleave(data & 0xF)
    ptr.store(_hex_table._dynamic_shuffle(nibbles))


@always_inline
fn _write_hex[amnt_hex_bytes: Int](p: UnsafePointer[Byte], decimal: Int):
    """Write a python compliant hexadecimal value into an uninitialized pointer
    location, assumed to be large enough for the value to be written.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from utils.write import _write_hex
    items = List[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0)
    alias S = StringSlice[__origin_of(items)]
    ptr = items.unsafe_ptr()
    _write_hex[8](ptr, ord("🔥"))
    assert_equal(r"\\U0001f525", S(ptr=ptr, length=10))
    memset_zero(ptr, len(items))
    _write_hex[4](ptr, ord("你"))
    assert_equal(r"\\u4f60", S(ptr=ptr, length=6))
    memset_zero(ptr, len(items))
    _write_hex[2](ptr, ord("Ö"))
    assert_equal(r"\\xd6", S(ptr=ptr, length=4))
    ```
    """

    constrained[amnt_hex_bytes in (2, 4, 8), "only 2 or 4 or 8 sequences"]()

    alias `\\` = Byte(ord("\\"))
    alias `x` = Byte(ord("x"))
    alias `u` = Byte(ord("u"))
    alias `U` = Byte(ord("U"))

    p.init_pointee_move(`\\`)

    @parameter
    if amnt_hex_bytes == 2:
        (p + 1).init_pointee_move(`x`)
        _hex_digits_to_hex_chars(p + 2, UInt8(decimal))
    elif amnt_hex_bytes == 4:
        (p + 1).init_pointee_move(`u`)
        _hex_digits_to_hex_chars(p + 2, UInt16(decimal))
    else:
        (p + 1).init_pointee_move(`U`)
        _hex_digits_to_hex_chars(p + 2, UInt32(decimal))
