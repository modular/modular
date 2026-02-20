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
"""Provides internal formatting utilities for the standard library.

This module contains helper types and functions used internally by the Mojo
standard library for formatting and writing data. These utilities are not
intended for public use and may change without notice.
"""

from builtin.constrained import _constrained_conforms_to
from io.io import _printf
from os import abort
from reflection.type_info import _unqualified_type_name
from sys import align_of, size_of
from sys.info import is_gpu
from sys.param_env import env_get_int

from bit import byte_swap
from memory import Span, bitcast, memcpy


fn constrained_conforms_to_writable[*Ts: AnyType, Parent: AnyType]():
    comptime for i in range(Variadic.size(Ts)):
        comptime T = Ts[i]
        _constrained_conforms_to[
            conforms_to(T, Writable),
            Parent=Parent,
            Element=T,
            ParentConformsTo="Writable",
        ]()


struct _SequenceWriter[W: Writer, origin: MutOrigin](Movable, Writer):
    """A writer that handles sequences of elements with separators.

    This writer ensures separators are only inserted between elements, even
    when an element's write_to() method calls write() multiple times.
    """

    var writer: Pointer[Self.W, Self.origin]
    var is_first_element: Bool
    var at_element_start: Bool
    var sep: StaticString

    fn __init__(out self, ref[Self.origin] writer: Self.W, sep: StaticString):
        self.writer = Pointer(to=writer)
        self.is_first_element = True
        self.at_element_start = True
        self.sep = sep

    @always_inline
    fn next_element(mut self):
        """Mark the start of the next element in the sequence."""
        self.at_element_start = True

    @always_inline
    fn write_string(mut self, string: StringSlice):
        self.write(string)

    @always_inline
    fn write[*Ts: Writable](mut self, *args: *Ts):
        if self.at_element_start:
            if not self.is_first_element:
                self.writer[].write_string(self.sep)
            self.is_first_element = False
            self.at_element_start = False

        comptime for i in range(args.__len__()):
            args[i].write_to(self.writer[])


# TODO (MOCO-2367): Use unified closures once they correctly capture parameters.
@always_inline
fn write_sequence_to[
    W: Writer, ElementFn: fn[T: Writer](mut T) raises StopIteration capturing
](
    mut writer: W,
    start: StaticString = "[",
    end: StaticString = "]",
    sep: StaticString = ", ",
):
    """Writes a sequence of elements to a writer using a callback function.

    This function writes elements by repeatedly calling the provided `ElementFn`
    callback until it raises `StopIteration`. Each element is separated by the
    specified separator, and the sequence is enclosed by opening and closing
    delimiters.

    This is useful for writing sequences where the elements are generated
    dynamically (e.g., from an iterator) rather than being known at compile time.

    Parameters:
        W: The writer type. Must conform to `Writer`.
        ElementFn: A callback function that writes a single element. It receives
            a mutable writer and should raise `StopIteration` when the sequence
            is exhausted.

    Args:
        writer: The writer to write to.
        start: The starting delimiter (default: `"["`).
        end: The ending delimiter (default: `"]"`).
        sep: The separator between elements (default: `", "`).
    """
    writer.write_string(start)

    var sequence_writer = _SequenceWriter(writer, sep)

    while True:
        try:
            ElementFn(sequence_writer)
            sequence_writer.next_element()
        except:
            break
    writer.write_string(end)


@always_inline
fn write_sequence_to[
    *Ts: Writable,
](
    mut writer: Some[Writer],
    *args: *Ts,
    start: StaticString,
    end: StaticString,
    sep: StaticString = ", ",
):
    """Writes a sequence of writable values to a writer with delimiters.

    This function formats a variadic list of writable values as a delimited
    sequence, such as `[item1, item2, item3]`.

    Parameters:
        Ts: Types of the values to write. Must conform to `Writable`.

    Args:
        writer: The writer to write to.
        args: The variadic list of values to write.
        start: The starting delimiter.
        end: The ending delimiter.
        sep: The separator between items (default: `", "`).
    """
    args._write_to(writer, start=start, end=end, sep=sep)


# TODO (MOCO-2367): Use unified closures once they correctly capture parameters.
@always_inline
fn write_sequence_to[
    size: Int,
    ElementFn: fn[i: Int](mut Some[Writer]) capturing,
](
    mut writer: Some[Writer],
    open: StaticString = "[",
    close: StaticString = "]",
    sep: StaticString = ", ",
):
    """Writes a compile-time sized sequence of elements using an indexed callback.

    This function writes a fixed number of elements determined at compile time by
    calling the provided callback function for each index from 0 to `size - 1`.
    Each element is separated by the specified separator, and the sequence is
    enclosed by opening and closing delimiters.

    This overload is useful when you have a compile-time known number of elements
    and need to generate each element based on its index.

    Parameters:
        size: The number of elements in the sequence (must be known at compile time).
        ElementFn: A callback function that writes a single element given its index.
            It receives a mutable writer and the index as a compile-time parameter.

    Args:
        writer: The writer to write to.
        open: The opening delimiter (default: `"["`).
        close: The closing delimiter (default: `"]"`).
        sep: The separator between elements (default: `", "`).
    """
    writer.write_string(open)

    comptime for i in range(size):
        comptime if i != 0:
            writer.write_string(sep)
        ElementFn[i=i](writer)

    writer.write_string(close)


@fieldwise_init
struct TypeNames[*Types: AnyType](ImplicitlyCopyable, Writable):
    """A wrapper type that writes a comma-separated list of type names."""

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        @parameter
        fn elements[i: Int](mut writer: Some[Writer]):
            writer.write_string(_unqualified_type_name[Self.Types[i]]())

        write_sequence_to[
            size = Variadic.size(Self.Types),
            ElementFn=elements,
        ](writer, open="", close="")


@always_inline
fn write_repr_to[T: AnyType](t: T, mut writer: Some[Writer]):
    comptime assert conforms_to(T, Writable), "T must be Writable"
    trait_downcast[Writable](t).write_repr_to(writer)


@always_inline
fn write_to[T: AnyType](t: T, mut writer: Some[Writer]):
    comptime assert conforms_to(T, Writable), "T must be Writable"
    trait_downcast[Writable](t).write_to(writer)


struct Repr[T: Writable, o: ImmutOrigin](ImplicitlyCopyable, Writable):
    """A wrapper type that writes the repr representation of a value.

    This struct wraps a reference to a `Writable` value and ensures that when
    the wrapper is written to a writer, it calls the underlying value's
    `write_repr_to` method instead of `write_to`. This allows formatting code
    to uniformly call `write_to` on the wrapper while still getting repr output.

    Parameters:
        T: The type of the value to wrap. Must conform to `Writable`.
        o: The immutable origin of the referenced value.
    """

    var _value: Pointer[Self.T, Self.o]

    @always_inline
    fn __init__(out self, ref[Self.o] value: Self.T):
        """Constructs a `Repr` wrapper around a reference to a value.

        Args:
            value: The value to wrap. A reference is stored, not a copy.
        """
        self._value = Pointer(to=value)

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes the repr representation of the wrapped value.

        This method delegates to the wrapped value's `write_repr_to` method,
        ensuring repr output when the wrapper is written.

        Args:
            writer: The writer to write to.
        """
        self._value[].write_repr_to(writer)


struct Named[T: Writable, o: ImmutOrigin](ImplicitlyCopyable, Writable):
    """A wrapper type that writes a named field in the format `name=value`.

    This struct is useful for formatting struct fields or named parameters,
    producing output like `field_name=42`. It wraps a reference to a value
    along with its name.

    Parameters:
        T: The type of the value. Must conform to `Writable`.
        o: The immutable origin of the referenced value.
    """

    var _name: StaticString
    var _value: Pointer[Self.T, Self.o]

    @always_inline
    fn __init__(out self, name: StaticString, ref[Self.o] value: Self.T):
        """Constructs a `Named` wrapper for a field.

        Args:
            name: The name of the field.
            value: The value of the field. A reference is stored, not a copy.
        """
        self._name = name
        self._value = Pointer(to=value)

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes the named field in the format `name=value`.

        Args:
            writer: The writer to write to.
        """
        writer.write(self._name, "=", self._value[])


struct FormatStruct[T: Writer, o: MutOrigin](Movable):
    """A helper type for formatting struct representations.

    This struct provides a fluent interface for writing struct-like
    representations in the format `StructName[param1, param2](field1, field2)`.
    It is designed to simplify the implementation of `write_to` and
    `write_repr_to` methods for user-defined types.

    The typical usage pattern is:
    1. Construct with the struct name (writes the name immediately)
    2. Optionally call `params()` to write type parameters in brackets
    3. Call `fields()` to write field values in parentheses

    Parameters:
        T: The writer type. Must conform to `Writer`.
        o: The mutable origin of the writer.
    """

    var _writer: Pointer[Self.T, Self.o]

    @always_inline
    fn __init__(out self, ref[Self.o] writer: Self.T, name: StaticString):
        """Constructs a `FormatStruct` and writes the struct name.

        Args:
            writer: The writer to write to.
            name: The name of the struct to write.
        """
        writer.write_string(name)
        self._writer = Pointer(to=writer)

    @always_inline
    fn params[*Ts: Writable](self, *args: *Ts) -> ref[self] Self:
        """Writes type parameters in bracket notation `[param1, param2, ...]`.

        This method is used to write compile-time parameters of a parameterized
        struct. It returns a reference to self to enable method chaining.

        Parameters:
            Ts: Types of the parameters to write. Must conform to `Writable`.

        Args:
            args: The parameter values to write.

        Returns:
            A reference to this `FormatStruct` instance for method chaining.
        """
        args._write_to(self._writer[], start="[", end="]")
        return self

    @always_inline
    fn fields[*Ts: Writable](self, *args: *Ts):
        """Writes field values in parentheses `(field1, field2, ...)`.

        This method is used to write the runtime field values of a struct.
        This should typically be the final call in a formatting chain, as it
        does not return a reference for further chaining.

        Parameters:
            Ts: Types of the fields to write. Must conform to `Writable`.

        Args:
            args: The field values to write.
        """
        args._write_to(self._writer[], start="(", end=")")

    # TODO (MOCO-2367): Use unified closures once they correctly capture parameters.
    @always_inline
    fn fields[FieldsFn: fn[T: Writer](mut T) capturing](self):
        """Writes field values in parentheses using a callback function.

        This overload is used when field values need to be generated dynamically
        (e.g., from an iterator) rather than being known at compile time. The
        callback function receives the writer and is responsible for writing all
        field content between the opening and closing parentheses.

        This is useful for writing collections or other types where the number of
        fields isn't known at compile time or where you need custom control over
        field formatting.

        Parameters:
            FieldsFn: A callback function that writes the field content. It
                receives a mutable writer.
        """
        self._writer[].write_string("(")
        FieldsFn(self._writer[])
        self._writer[].write_string(")")


comptime HEAP_BUFFER_BYTES = env_get_int["HEAP_BUFFER_BYTES", 2048]()
"""How much memory to pre-allocate for the heap buffer, will abort if exceeded."""

comptime STACK_BUFFER_BYTES = UInt(env_get_int["STACK_BUFFER_BYTES", 4096]())
"""The size of the stack buffer for IO operations from CPU."""


struct _WriteBufferHeap(Writable, Writer):
    var _data: UnsafePointer[Byte, MutExternalOrigin]
    var _pos: Int

    fn __init__(out self):
        comptime alignment: Int = align_of[Byte]()
        self._data = __mlir_op.`pop.stack_allocation`[
            count = HEAP_BUFFER_BYTES._mlir_value,
            _type = type_of(self._data)._mlir_type,
            alignment = alignment._mlir_value,
        ]()
        self._pos = 0

    fn write_list[
        T: Copyable & Writable, //
    ](mut self, values: List[T, ...], *, sep: StaticString = StaticString()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    # TODO: Removing @always_inline causes some AMD tests to fail.
    # This is likely because not inlining causes _WriteBufferHeap to
    # add a conditional allocation branch which is not supported on AMD.
    # However, when its inlined, the branch (and allocation) are removed.
    # We should consider uses _WriteBufferStack on AMD instead.
    @always_inline
    fn write_string(mut self, string: StringSlice):
        var len_bytes = string.byte_length()
        if len_bytes + self._pos > HEAP_BUFFER_BYTES:
            _printf[
                "HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D"
                " HEAP_BUFFER_BYTES=4096`\n"
            ]()
            abort()
        memcpy(
            dest=self._data + self._pos,
            src=string.unsafe_ptr(),
            count=len_bytes,
        )
        self._pos += len_bytes

    fn write_to(self, mut writer: Some[Writer]):
        writer.write_string(
            StringSlice(unsafe_from_utf8=Span(ptr=self._data, length=self._pos))
        )

    fn nul_terminate(mut self):
        if self._pos + 1 > HEAP_BUFFER_BYTES:
            _printf[
                "HEAP_BUFFER_BYTES exceeded, increase with: `mojo -D"
                " HEAP_BUFFER_BYTES=4096`\n"
            ]()
            abort()
        self._data[self._pos] = 0
        self._pos += 1

    fn as_string_slice[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self) -> StringSlice[origin]:
        return StringSlice(
            unsafe_from_utf8=Span(
                ptr=self._data.mut_cast[mut]().unsafe_origin_cast[origin](),
                length=self._pos,
            )
        )


struct _WriteBufferStack[
    origin: MutOrigin,
    W: Writer,
    //,
    stack_buffer_bytes: UInt = STACK_BUFFER_BYTES,
](Writer):
    var data: InlineArray[UInt8, Int(Self.stack_buffer_bytes)]
    var pos: Int
    var writer: Pointer[Self.W, Self.origin]

    fn __init__(out self, ref[Self.origin] writer: Self.W):
        self.data = InlineArray[UInt8, Int(Self.stack_buffer_bytes)](
            uninitialized=True
        )
        self.pos = 0
        self.writer = Pointer(to=writer)

    fn write_list[
        T: Copyable & Writable, //
    ](mut self, values: List[T, ...], *, sep: String = String()):
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    fn flush(mut self):
        self.writer[].write_string(
            StringSlice(
                unsafe_from_utf8=Span(
                    ptr=self.data.unsafe_ptr(), length=self.pos
                )
            )
        )
        self.pos = 0

    fn write_string(mut self, string: StringSlice):
        len_bytes = string.byte_length()
        # If span is too large to fit in buffer, write directly and return
        if len_bytes > Int(Self.stack_buffer_bytes):
            self.flush()
            self.writer[].write_string(string)
            return
        # If buffer would overflow, flush writer and reset pos to 0.
        elif self.pos + len_bytes > Int(Self.stack_buffer_bytes):
            self.flush()
        # Continue writing to buffer
        memcpy(
            dest=self.data.unsafe_ptr() + self.pos,
            src=string.unsafe_ptr(),
            count=len_bytes,
        )
        self.pos += len_bytes


struct _TotalWritableBytes(Writer):
    var size: Int

    fn __init__(out self):
        self.size = 0

    fn __init__[
        T: Copyable & Writable,
        //,
        origin: ImmutOrigin,
    ](
        out self,
        values: Span[T, ...],
        sep: StringSlice[origin] = StringSlice[origin](),
    ):
        self.size = 0
        var length = len(values)
        if length == 0:
            return
        self.write(values[0])
        if length > 1:
            for i in range(1, length):
                self.write(sep, values[i])

    fn write_string(mut self, string: StringSlice):
        self.size += string.byte_length()


fn _ord_ascii(s: StringSlice) -> UInt8:
    return UInt8(ord(s))


# fmt: off
comptime _hex_table = SIMD[DType.uint8, 16](
    _ord_ascii("0"), _ord_ascii("1"), _ord_ascii("2"), _ord_ascii("3"), _ord_ascii("4"),
    _ord_ascii("5"), _ord_ascii("6"), _ord_ascii("7"), _ord_ascii("8"), _ord_ascii("9"),
    _ord_ascii("a"), _ord_ascii("b"), _ord_ascii("c"), _ord_ascii("d"), _ord_ascii("e"), _ord_ascii("f"),
)
# fmt: on


@always_inline
fn _hex_digits_to_hex_chars(
    decimal: Scalar,
) -> SIMD[DType.uint8, size_of[decimal.dtype]() * 2]:
    """Return a fixed width hexadecimal value according to the scalar dtype.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from io.write import _hex_digits_to_hex_chars
    items: List[Byte] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    comptime S = StringSlice[origin_of(items)]
    ptr = items.unsafe_ptr()
    ptr.store(_hex_digits_to_hex_chars(UInt32(ord("🔥"))))
    assert_equal("0001f525", S(ptr=ptr, length=8))
    ptr.store(_hex_digits_to_hex_chars(UInt16(ord("你"))))
    assert_equal("4f60", S(ptr=ptr, length=4))
    ptr.store(_hex_digits_to_hex_chars(UInt8(ord("Ö"))))
    assert_equal("d6", S(ptr=ptr, length=2))
    ```
    """
    comptime size = size_of[decimal.dtype]()
    var bytes = bitcast[DType.uint8, size](byte_swap(decimal))
    var nibbles = (bytes >> 4).interleave(bytes & 0xF)
    return _hex_table._dynamic_shuffle(nibbles)


@always_inline
fn _write_hex[
    *, amnt_hex_bytes: Int
](mut writer: Some[Writer], decimal: Scalar):
    """Write a python compliant hexadecimal value into a writer.

    Examples:

    ```mojo
    %# from memory import memset_zero
    %# from testing import assert_equal
    %# from utils import StringSlice
    %# from io.write import _write_hex
    var s = String()
    _write_hex[amnt_hex_bytes=8](s, ord("🔥"))
    assert_equal("\\U0001f525", s)
    s = ""
    _write_hex[amnt_hex_bytes=4](s, ord("你"))
    assert_equal("\\u4f60", s)
    s = ""
    _write_hex[amnt_hex_bytes=2](s, ord("Ö"))
    assert_equal("\\xd6", s)
    ```
    """

    comptime assert amnt_hex_bytes in (2, 4, 8), "only 2 or 4 or 8 sequences"

    comptime `\\` = Byte(ord("\\"))
    comptime `x` = Byte(ord("x"))
    comptime `u` = Byte(ord("u"))
    comptime `U` = Byte(ord("U"))

    comptime if amnt_hex_bytes == 2:
        var chars = _hex_digits_to_hex_chars(UInt8(decimal))
        var buf = InlineArray[Byte, 4](uninitialized=True)
        buf[0] = `\\`
        buf[1] = `x`
        (buf.unsafe_ptr() + 2).store(chars)
        writer.write_string(StringSlice(unsafe_from_utf8=Span(buf)))
    elif amnt_hex_bytes == 4:
        var chars = _hex_digits_to_hex_chars(UInt16(decimal))
        var buf = InlineArray[Byte, 6](uninitialized=True)
        buf[0] = `\\`
        buf[1] = `u`
        (buf.unsafe_ptr() + 2).store(chars)
        writer.write_string(StringSlice(unsafe_from_utf8=Span(buf)))
    else:
        var chars = _hex_digits_to_hex_chars(UInt32(decimal))
        var buf = InlineArray[Byte, 10](uninitialized=True)
        buf[0] = `\\`
        buf[1] = `U`
        (buf.unsafe_ptr() + 2).store(chars)
        writer.write_string(StringSlice(unsafe_from_utf8=Span(buf)))
