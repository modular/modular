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
from std.collections.string.format import _FormatUtils, _comptime_list_to_span
from std.sys import is_compile_time
from std.utils import Variant
import std.format._utils as fmt


@always_inline
fn _strlen(ptr: UnsafePointer[mut=False, Byte]) -> Int:
    var offset = 0
    while ptr[offset]:
        offset += 1
    return offset


struct TString[
    origins: ImmutOrigin, //, format_string: StaticString, *Ts: Writable
](Movable, Writable):
    """A template string that captures interpolated values at compile-time.

    TString is a zero-cost abstraction for string interpolation that preserves
    type information and defers formatting until explicitly requested. Unlike
    regular strings or f-strings, TString retains the original format template
    and typed values, enabling efficient lazy formatting and type-safe string
    composition.

    TString instances are created by the compiler when using t-string literal
    syntax: `t"Hello {name}!"`.

    Parameters:
        origins: The origin of the interpolated values.
        format_string: The compile-time format string template.
        Ts: The types of the interpolated values.
    """

    comptime _InjectedValues = VariadicPack[
        origin = Self.origins, False, Writable, *Self.Ts
    ]
    var _values: Self._InjectedValues

    @doc_private
    @always_inline
    fn __init__(out self, *, var pack: Self._InjectedValues):
        self._values = pack^

    @always_inline
    fn _write_to_impl(
        self, mut writer: Some[Writer], encoded_bytes: Span[mut=False, Byte]
    ):
        var offset = 0

        @always_inline
        fn write_string() unified {
            read encoded_bytes, read offset, mut writer
        } -> Int:
            var literal_start = encoded_bytes.unsafe_ptr() + offset
            var literal_length = _strlen(literal_start)
            var string_literal = StringSlice(
                ptr=literal_start, length=literal_length
            )
            writer.write_string(string_literal)
            return literal_length

        # Alternate writing NUL terminated string-literal part, followed
        # by the interpolated replacement field.
        comptime for i in range(Variadic.size(Self.Ts)):
            var length = write_string()
            offset += length + 1
            self._values[i].write_to(writer)

        # Write the final string literal part.
        _ = write_string()

    fn write_to(self, mut writer: Some[Writer]):
        """Write the formatted string to a writer.

        This method implements the `Writable` trait by formatting the TString's
        template with its interpolated values and writing the result to the
        provided writer. The format string is compiled at compile-time for
        optimal performance.

        Args:
            writer: The writer to output the formatted string to.
        """
        comptime bytes = _encode_format_string_comptime[Self.format_string]()
        if is_compile_time():
            self._write_to_impl(writer, materialize[bytes]())
        else:
            var span = _comptime_list_to_span[bytes]()
            self._write_to_impl(writer, span)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write a debug representation of the TString to a writer.

        This method provides a detailed view of the TString's internal structure,
        showing the format template, type parameters, and the actual interpolated
        values. This is useful for debugging and understanding the TString's
        composition.

        Args:
            writer: The writer to output the debug representation to.
        """

        @parameter
        fn fields(mut writer: Some[Writer]):
            self._values._write_to[is_repr=True](writer, start="", end="")

        fmt.FormatStruct(writer, "TString").params(
            fmt.Repr(self.format_string),
            fmt.TypeNames[*Self.Ts](),
        ).fields[FieldsFn=fields]()


@always_inline
fn __make_tstring[
    format_string: __mlir_type.`!kgen.string`, *Ts: Writable
](
    *args: *Ts,
    out tstring: TString[
        origins = ImmutOrigin(type_of(args).origin),
        StaticString(format_string),
        *Ts,
    ],
):
    """Compiler entry point for creating TStrings from t-string expressions.

    This function is called by the compiler when it encounters a t-string
    literal expression like `t"Hello {name}!"`. The compiler extracts the
    format string and argument expressions, then generates a call to this
    function to construct the corresponding TString object.

    Parameters:
        format_string: The compile-time string literal containing the template.
        Ts: The types of the interpolated values.

    Args:
        args: The values to interpolate into the template string.

    Returns:
        The constructed TString object.
    """
    tstring = {pack = rebind_var[type_of(tstring)._InjectedValues](args.copy())}


fn _encode_format_string_comptime[format: StringSlice]() -> List[Byte]:
    comptime result = _encode_format_string_no_raises(format)
    comptime if result.isa[Error]():
        comptime assert False, String(result[Error])
    else:
        return result.take[List[Byte]]()


fn _encode_format_string_no_raises(
    format: StringSlice,
) -> Variant[List[Byte], Error]:
    try:
        return _encode_format_string(format)
    except e:
        return e^


def _encode_format_string(format: StringSlice) raises -> List[Byte]:
    """Encode a format string into a flat byte sequence.

    The output is an alternating sequence of NUL-terminated literal segments
    and replacement field boundaries. For N replacement fields, there are
    always N+1 literal segments.

    The replacement fields themselves are not stored — their positions are
    implied by the NUL boundaries. Escaped braces (`{{`/`}}`) are resolved
    to `{`/`}` in the literal text.

    If the format string starts with `{}`, the first literal segment is
    empty (a bare NUL byte). Likewise if the format string ends with `{}`,
    the last literal segment is empty. This means the output always begins
    and ends with a (possibly empty) NUL-terminated literal segment.

    For example, `"result: {} + {} = {}"` encodes as
    `"result: \0 + \0 = \0\0"`, which we walks through as:

        1. literal: "result: \0"
        2. arg: 0
        3. literal: " + \0"
        4. arg: 1
        5. literal: " = \0"
        6. arg: 2
        7. literal: "\0"      (empty — format ends with {})

    At runtime, the we write bytes until NUL to get a literal
    segment, writes the next interpolated argument, and repeats until
    the final literal segment (which has no argument after it).

    Args:
        format: The format string to encode.

    Returns:
        A list of bytes containing the encoded format string.

    Raises:
        If the format string contains non-empty replacement fields, unmatched
        braces, or other syntax errors.
    """
    comptime LBRACE = Byte(ord("{"))
    comptime RBRACE = Byte(ord("}"))
    comptime NUL = Byte(0)

    var result = List[Byte]()
    var bytes = format.as_bytes()
    var i = 0

    @always_inline
    fn peek_next_is(byte: Byte) unified {read} -> Bool:
        return i + 1 < len(bytes) and bytes[i + 1] == byte

    while i < len(bytes):
        var byte = bytes[i]
        if byte == LBRACE:
            if peek_next_is(LBRACE):
                # Escaped brace {{ -> {
                result.append(LBRACE)
            elif peek_next_is(RBRACE):
                # Empty replacement field {} -> NUL separator.
                result.append(NUL)
            else:
                raise Error(
                    "unclosed/non-empty replacement field in format string"
                )

            # skip past escaped brace or replacement field
            i += 2
        elif byte == RBRACE:
            if not peek_next_is(RBRACE):
                raise Error("single '}' is not allowed in format string")

            # Escaped brace }} -> }
            result.append(RBRACE)
            i += 2
        else:
            result.append(byte)
            i += 1

    # Terminate the final literal segment with NUL.
    result.append(NUL)
    return result^
