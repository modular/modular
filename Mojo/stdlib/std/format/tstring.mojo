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
"""Implements `TString`, a template string that captures interpolated values at compile-time."""
import std.format._utils as fmt
from std.builtin.globals import global_constant
from std.os import abort
from std.utils import StaticTuple


struct _ErasedWriter[origin: MutOrigin](TrivialRegisterPassable, Writer):
    """A type-erased `Writer`."""

    var _erased: Pointer[NoneType, Self.origin]
    var _write_string: def(MutPointer[NoneType, _], StringSpan[_]) thin

    def __init__[T: Writer](out self, ref[Self.origin] writer: T):
        def write_string(
            erased: MutPointer[NoneType, _],
            string: StringSpan[_],
        ):
            erased.unsafe_bitcast[T]()[].write_string(string)

        self._erased = Pointer(to=writer).unsafe_bitcast[NoneType]()
        self._write_string = write_string

    def write_string(mut self, string: StringSpan[_]):
        self._write_string(self._erased, string)


struct _FormatArgument[origin: ImmOrigin](TrivialRegisterPassable):
    """A type-erased `Writable` value.

    Parameters:
        origin: The origin of the erased value.
    """

    var _erased: Pointer[NoneType, Self.origin]
    var _write_to: def(ImmPointer[NoneType, _], var _ErasedWriter[_]) thin

    def __init__[
        T: Writable
    ](out self, ref[Self.origin] writable: T,):
        """Captures a reference to an interpolated value.

        Args:
            writable: The value to reference.
        """

        def write_to(
            data: ImmPointer[NoneType, _], var writer: _ErasedWriter[_]
        ):
            data.unsafe_bitcast[T]()[].write_to(writer)

        self._erased = Pointer(to=writable).unsafe_bitcast[NoneType]()
        self._write_to = write_to

    @inline(.always)
    def dispatch_write_to(self, writer: _ErasedWriter[_]):
        self._write_to(self._erased, writer)


struct TString[origins: ImmOrigin, array_origin: ImmOrigin](
    RegisterPassable, Writable
):
    """A template string that captures interpolated values at compile-time.

    TString is a zero-cost abstraction for string interpolation that preserves
    type information and defers formatting until explicitly requested. Unlike
    regular strings or f-strings, TString retains the original format template
    and typed values, enabling efficient lazy formatting and type-safe string
    composition.

    TString instances are created by the compiler when using t-string literal
    syntax: `t"Hello {name}!"`.

    Parameters:
        origins: The union of the origins of the interpolated values.
        array_origin: The origin of the array holding those values, which
            the t-string expression owns and this `TString` borrows.
    """

    comptime _ArgumentSpan = ImmSpan[
        _FormatArgument[Self.origins], Self.array_origin
    ]

    var _values: Self._ArgumentSpan
    """Type-erased references, one per replacement field in the template."""
    var _encoded: ImmPointer[Byte, ImmStaticOrigin]
    """The template's length-prefixed literal parts, encoded at construction."""

    @doc_hidden
    @inline(.always)
    def __init__(
        out self,
        *,
        values: Self._ArgumentSpan,
        encoded: ImmPointer[Byte, ImmStaticOrigin],
    ):
        self._values = values
        self._encoded = encoded

    def _write_to_impl(self, var writer: _ErasedWriter[_]):
        """Render the template into an already type-erased writer.

        `write_to` specializes per writer type; this does not, so the walk
        over the encoded template can be shared. Marking it `@inline(.never)`
        is what actually shares it — LLVM inlines it by default — which costs
        a call per `write_to` and is not currently worth it.
        """
        var offset = 0

        @inline(.always)
        def write_string() {mut writer, imm} -> Int:
            var literal_length = Int(self._encoded[unsafe_offset=offset])
            var literal_start = self._encoded.unsafe_offset(offset + 1)
            var string_literal = StringSpan(
                unsafe_from_utf8=Span(
                    unsafe_ptr=literal_start, length=literal_length
                )
            )
            writer.write_string(string_literal)
            return literal_length

        # Alternate writing a length-prefixed string-literal part, followed
        # by the interpolated replacement field.
        for argument in self._values:
            var length = write_string()
            offset += length + 1
            argument.dispatch_write_to(writer)

        # Write the final string literal part.
        _ = write_string()

    def write_to(self, mut writer: Some[Writer]):
        """Write the formatted string to a writer.

        This method implements the `Writable` trait by formatting the TString's
        template with its interpolated values and writing the result to the
        provided writer.

        Args:
            writer: The writer to output the formatted string to.
        """
        var erased = _ErasedWriter(writer)
        self._write_to_impl(erased)

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write a debug representation of the TString to a writer.

        The interpolated values are type-erased, so neither their types nor
        their values are recoverable here; use `write_to` to render them.

        Args:
            writer: The writer to output the debug representation to.
        """
        fmt.FormatStruct(writer, "TString").fields()


# Each literal segment's length is stored in a single byte.
comptime _MAX_LITERAL_SEGMENT_LENGTH = 255


@inline(.always)
def __make_tstring[
    format_string: __mlir_type.`!kgen.string`,
    origins: ImmOrigin,
](
    ref array: Array[_FormatArgument[origins], _],
    out tstring: TString[origins, origin_of(array)],
):
    """Compiler entry point for creating TStrings from t-string expressions.

    For `t"Hello {name}!"` the compiler emits the array as a list literal at
    the t-string itself, so it lands in the frame that owns the interpolated
    values and outlives the borrowing `TString`. The values arrive already
    wrapped in `_FormatArgument`, formed at the t-string too so each one
    records the value's real address; a variadic pack of `Ts` would instead let
    the ABI promote register-passable arguments to by-value copies. Nothing
    here is parameterized on those types, so t-strings sharing a template share
    this specialization.

    Parameters:
        format_string: The compile-time string literal containing the template.
        origins: The union of the origins of the interpolated values.

    Args:
        array: The interpolated values, one per replacement field.

    Returns:
        The constructed TString object.
    """
    comptime fmt_str = StaticString(format_string)
    comptime longest_literal = _longest_literal_segment(fmt_str)
    comptime assert longest_literal <= _MAX_LITERAL_SEGMENT_LENGTH, String(
        "t-string literal segment is ",
        longest_literal,
        " bytes long; at most ",
        _MAX_LITERAL_SEGMENT_LENGTH,
        (
            " bytes of literal text are supported without a replacement field"
            " in between"
        ),
    )
    comptime length = _count_encoded_bytes(fmt_str)
    comptime bytes = _encoded_bytes[length](fmt_str)

    ref global_bytes = global_constant[bytes]()
    tstring = {
        values = Span(array),
        encoded = Pointer(to=global_bytes).unsafe_bitcast[Byte](),
    }


def _count_encoded_bytes(format: StaticString) -> Int:
    var count = 0

    def addone(_byte: Byte) {mut}:
        count += 1

    _encode_format_string(format, addone)
    return count


def _encoded_bytes[
    length: Int
](format: StaticString) -> StaticTuple[Byte, length]:
    var bytes = StaticTuple[Byte, length](fill=0)
    var index = 0

    def append(byte: Byte) {mut}:
        bytes[index] = byte
        index += 1

    _encode_format_string(format, append)
    return bytes


def _literal_segment(format: StaticString, start: Int) -> Tuple[Int, Int]:
    """Measures the literal segment of a format string that starts at `start`.

    Assumes well-formed braces; `_encode_format_string` rejects malformed ones.

    Args:
        format: The format string.
        start: The byte offset where the segment starts.

    Returns:
        The segment's length in bytes once escaped braces are resolved (`{{`
        and `}}` count as one byte each), and the byte offset where the
        segment ends: the `{}` that closes it, or the end of `format`.
    """
    comptime LBRACE = Byte(123)  # '{'
    comptime RBRACE = Byte(125)  # '}'

    var bytes = format.as_bytes()
    var i = start
    var length = 0
    while i < len(bytes):
        var byte = bytes.unsafe_ptr()[unsafe_offset=i]
        if byte == LBRACE or byte == RBRACE:
            if (
                byte == LBRACE
                and i + 1 < len(bytes)
                and bytes.unsafe_ptr()[unsafe_offset=i + 1] == RBRACE
            ):
                break
            i += 2
        else:
            i += 1
        length += 1
    return length, i


def _longest_literal_segment(format: StaticString) -> Int:
    """Returns the length in bytes of the longest literal segment of `format`.

    Args:
        format: The format string.

    Returns:
        The largest length `_literal_segment` reports for any segment.
    """
    var longest = 0
    var start = 0
    while True:
        var length, end = _literal_segment(format, start)
        longest = max(longest, length)
        if end >= format.byte_length():
            return longest
        start = end + 2  # Skip the `{}` that ends this segment.


def _encode_format_string(format: StaticString, f: Some[def(Byte)]):
    """Encode a format string into a flat byte sequence.

    The output is a sequence of length-prefixed literal segments: one byte
    holding the segment's length, followed by that many bytes of literal
    text. For N replacement fields, there are always N+1 literal segments.

    The replacement fields themselves are not stored — each one sits between
    two consecutive segments. Escaped braces (`{{`/`}}`) are resolved to
    `{`/`}` in the literal text. Because segments carry their length instead
    of ending in a NUL byte, the literal text may itself contain NUL bytes.

    If the format string starts with `{}`, the first literal segment is
    empty (a single length byte of 0). Likewise if the format string ends
    with `{}`, the last literal segment is empty.

    A literal segment longer than `_MAX_LITERAL_SEGMENT_LENGTH` (255) bytes
    does not fit in its length byte; `__make_tstring` rejects such a template
    at compile time.

    For example, `"result: {} + {} = {}"` encodes as the bytes
    `[8, "result: ", 3, " + ", 3, " = ", 0]`, which we walk through as:

        1. literal: "result: "  (length 8)
        2. arg: 0
        3. literal: " + "       (length 3)
        4. arg: 1
        5. literal: " = "       (length 3)
        6. arg: 2
        7. literal: ""          (length 0 — format ends with {})

    At runtime, we read a segment's length byte, write that many bytes,
    write the next interpolated argument, and repeat until the final literal
    segment (which has no argument after it).

    Args:
        format: The format string to encode.
        f: Called once per encoded byte, in order, instead of returning a
            buffer directly. This lets callers collect the bytes however
            they need to, for example counting them or writing them into a
            fixed-size buffer.
    """
    comptime LBRACE = Byte(123)  # '{'
    comptime RBRACE = Byte(125)  # '}'

    var bytes = format.as_bytes()
    var i = 0

    # Note: using `bytes.unsafe_ptr()[unsafe_offset=i]` is intentional over
    # using `bytes[i]` as it puts less stress on the comptime interpreter
    # resulting in better compile times.

    @inline(.always)
    def peek_next_is(byte: Byte) {imm} -> Bool:
        return (
            i + 1 < len(bytes)
            and bytes.unsafe_ptr()[unsafe_offset=i + 1] == byte
        )

    var first_length, _ = _literal_segment(format, 0)
    f(Byte(first_length))

    while i < len(bytes):
        var byte = bytes.unsafe_ptr()[unsafe_offset=i]
        if byte == LBRACE:
            if peek_next_is(LBRACE):
                # Escaped brace {{ -> {
                f(LBRACE)
            elif peek_next_is(RBRACE):
                # Empty replacement field {} -> length of the next segment.
                var next_length, _ = _literal_segment(format, i + 2)
                f(Byte(next_length))
            else:
                abort()

            # skip past escaped brace or replacement field
            i += 2
        elif byte == RBRACE:
            if not peek_next_is(RBRACE):
                abort()

            # Escaped brace }} -> }
            f(RBRACE)
            i += 2
        else:
            f(byte)
            i += 1
