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
"""String formatting utilities for Mojo.

This module provides string formatting functionality similar to Python's
`str.format()` method. The `format()` method (available on the
[`String`](/mojo/std/collections/string/string/String#format) and
[`StringSlice`](/mojo/std/collections/string/string_slice/StringSlice#format)
types) takes the current string as a template (or "format string"), which can
contain literal text and/or replacement fields delimited by curly braces (`{}`).
The replacement fields are replaced with the values of the arguments.

Replacement fields can mapped to the arguments in one of two ways:

- Automatic indexing by argument position:

  ```mojo
  var s = "{} is {}".format("Mojo", "🔥")
  ```

- Manual indexing by argument position:

  ```mojo
  var s = "{1} is {0}".format("hot", "🔥")
  ```

The replacement fields can also contain the `!r` or `!s` conversion flags, to
indicate whether the argument should be formatted using `repr()` or `String()`,
respectively:

```mojo
%# var some_object = String()
var s = "{!r}".format(some_object)
```

Note that the following features from Python's `str.format()` are
**not yet supported**:

- Named arguments (for example `"{name} is {adjective}"`).
- Accessing the attributes of an argument value (for example, `"{0.name}"`.
- Accessing an indexed value from the argument (for example, `"{1[0]}"`).
- Format specifiers for controlling output format (width, precision, and so on).

Examples:

```mojo
# Basic formatting
var s1 = "Hello {0}!".format("World")  # Hello World!

# Multiple arguments
var s2 = "{0} plus {1} equals {2}".format(1, 2, 3)  # 1 plus 2 equals 3

# Conversion flags
var s4 = "{!r}".format("test")  # "'test'"
```

This module has no public API; its functionality is available through the
[`String.format()`](/mojo/std/collections/string/string/String#format) and
[`StringSlice.format()`](/mojo/std/collections/string/string_slice/StringSlice#format)
methods.
"""


from std.builtin.globals import global_constant
from std.builtin.variadics import Variadic
from std.collections.string.string_slice import get_static_string
from std.compile import get_type_name
from std.utils import Variant

# ===-----------------------------------------------------------------------===#
# Formatter
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _PrecompiledEntries[
    format_origin: ImmutOrigin, entry_origin: ImmutOrigin, //, *Ts: Writable
](ImplicitlyCopyable):
    """Holds a non-owning view of precompiled format string entries.

    This struct stores the parsed replacement fields from a format string along
    with metadata for efficient formatting. It uses a `Span` to reference the
    entries without owning them, allowing it to work with both static (compile-
    time) and runtime-allocated entries:

    - **Static origins**: When used with `compile_entries_comptime()`, both the
      format string and entries have static storage duration (stored in global
      constants), enabling zero-allocation formatting.
    - **Runtime origins**: When converted from `_PrecompiledEntriesRuntime`, it
      provides a non-owning view of runtime-allocated entries via `Span`.

    Parameters:
        format_origin: The origin of the format string data (can be static or runtime).
        entry_origin: The origin of the entries array (can be static or runtime).
        Ts: The types of the arguments that will be formatted.
    """

    var entries: Span[_FormatCurlyEntry[Self.format_origin], Self.entry_origin]
    var size_hint: Int
    var format: StringSlice[Self.format_origin]


@fieldwise_init
struct _PrecompiledEntriesRuntime[
    format_origin: ImmutOrigin, //, *Ts: Writable
](Movable):
    """Holds precompiled format string entries with owned runtime-allocated storage.

    This struct is similar to `_PrecompiledEntries` but uses a `List` to own
    the dynamically-allocated entries. It's used by `compile_entries_runtime()`
    when parsing format strings at runtime. The entries can then be converted
    to a `_PrecompiledEntries` (via `Span`) for use with `format_precompiled()`.

    Parameters:
        format_origin: The origin of the format string data (runtime).
        Ts: The types of the arguments that will be formatted.
    """

    var entries: List[_FormatCurlyEntry[Self.format_origin]]
    var size_hint: Int
    var format: StringSlice[Self.format_origin]


@always_inline
fn _comptime_list_to_span[
    T: ImplicitlyDestructible & Copyable, //, list: List[T]
]() -> Span[T, StaticConstantOrigin]:
    """Convert a comptime list to a runtime span of static constant origin."""

    fn list_to_array[list: List[T]]() -> InlineArray[T, len(list)]:
        var array = InlineArray[T, len(list)](uninitialized=True)

        comptime for i in range(len(list)):
            UnsafePointer(to=array[i]).init_pointee_copy(materialize[list]()[i])
        return array^

    comptime array = list_to_array[list]()
    return Span(global_constant[array]())


comptime _FormatArgs = VariadicPack[element_trait=Writable, ...]


struct _FormatUtils:
    # TODO: Allow a way to provide a `comptime _PrecompiledEntries` to avoid
    # allocations in the `_PrecompiledEntries` struct.
    @staticmethod
    fn format_precompiled[
        *Ts: Writable,
    ](
        mut writer: Some[Writer],
        compiled: _PrecompiledEntries[*Ts],
        args: VariadicPack[_, Writable, *Ts],
    ):
        """Format the arguments using the given format string and precompiled entries.
        """
        comptime len_pos_args = type_of(args).__len__()
        var offset = 0
        var ptr = compiled.format.unsafe_ptr()
        var fmt_len = compiled.format.byte_length()

        @always_inline
        fn _build_slice(
            p: UnsafePointer[mut=False, UInt8], start: Int, end: Int
        ) -> StringSlice[p.origin]:
            return StringSlice(ptr=p + start, length=end - start)

        var auto_arg_index = 0
        for e in compiled.entries:
            # offset can equal fmt_len when format ends with a replacement field
            debug_assert(offset <= fmt_len, "offset > format.byte_length()")
            writer.write(_build_slice(ptr, offset, e.first_curly))
            e._format_entry[len_pos_args](writer, args, auto_arg_index)
            offset = e.last_curly + 1

        writer.write(_build_slice(ptr, offset, fmt_len))

    @staticmethod
    fn format(
        format: StringSlice, args: VariadicPack[element_trait=Writable, ...]
    ) raises -> String:
        """Format the arguments using the given format string."""
        var buffer = String()
        Self.format_to_runtime(buffer, format, args)
        return buffer^

    @staticmethod
    fn format_to_runtime(
        mut writer: Some[Writer],
        format: StringSlice,
        args: VariadicPack[_, Writable, ...],
    ) raises:
        """Format arguments into a writer using a runtime format string.

        This function parses and compiles the format string at runtime, then
        writes the formatted output to the provided writer. Use this when the
        format string is not known at compile time.

        For compile-time format strings, prefer `format_to_comptime()` which
        parses the format string at compile time and can catch format errors
        during compilation.

        Args:
            writer: The writer to write the formatted output to.
            format: The format string to parse.
            args: The arguments to format into the replacement fields.

        Raises:
            An error if the format string is invalid or if replacement fields
            don't match the provided arguments.
        """
        comptime Ts = type_of(args).element_types
        var compiled = Self.compile_entries_runtime[*Ts](format)
        Self.format_precompiled(
            writer=writer,
            compiled=_PrecompiledEntries[*Ts](
                Span(compiled.entries), compiled.size_hint, compiled.format
            ),
            args=args,
        )

    @staticmethod
    fn format_to_comptime[
        format: StaticString
    ](mut writer: Some[Writer], args: VariadicPack[_, Writable, ...]):
        """Format arguments into a writer using a compile-time format string.

        This function parses and compiles the format string at compile time,
        enabling zero-allocation formatting and catching format errors during
        compilation rather than at runtime.

        This is more efficient than `format_to_runtime()` because:
        - Format string parsing happens once at compile time
        - Format errors are caught during compilation
        - The compiled entries are stored in static memory
        - No runtime allocations for parsing the format string

        Parameters:
            format: The format string to parse at compile time. Must be a
                string literal or StaticString.

        Args:
            writer: The writer to write the formatted output to.
            args: The arguments to format into the replacement fields.
        """
        comptime Ts = type_of(args).element_types

        comptime result = _FormatUtils.compile_entries_runtime_no_raises[*Ts](
            format
        )

        comptime if result.isa[Error]():
            comptime assert not result.isa[Error](), String(result[Error])
        else:
            comptime entries = result[type_of(result).Ts[0]]
            _FormatUtils.format_precompiled[*Ts](
                writer,
                _PrecompiledEntries[*Ts](
                    _comptime_list_to_span[entries.entries](),
                    entries.size_hint,
                    get_static_string[format](),
                ),
                args,
            )

    @staticmethod
    fn compile_entries_runtime_no_raises[
        *Ts: Writable
    ](
        format: StringSlice,
    ) -> Variant[
        _PrecompiledEntriesRuntime[
            format_origin = ImmutOrigin(format.origin), *Ts
        ],
        Error,
    ]:
        """Parses and compiles a format string without raising an error.

        Instead of raising an error, this function returns a `Variant`. This is
        useful if you are trying to call `compile_entries` at comptime which
        does not support raising functions.
        """
        try:
            return Self.compile_entries_runtime[*Ts](format)
        except e:
            return e^

    @staticmethod
    fn compile_entries_runtime[
        *Ts: Writable
    ](
        format: StringSlice,
    ) raises -> _PrecompiledEntriesRuntime[
        format_origin = ImmutOrigin(format.origin), *Ts
    ]:
        """Parses and compiles a format string at runtime.

        This function analyzes a format string, parses all replacement fields
        (e.g., `{}`, `{0}`, `{!r}`), and validates them against the provided
        argument types. The parsed entries are stored in a dynamically-allocated
        `List` for later use in formatting.

        Parameters:
            Ts: The types of the arguments that will be formatted.

        Args:
            format: The format string to parse.

        Returns:
            A `_PrecompiledEntriesRuntime` struct containing the parsed format
            entries and metadata.

        Raises:
            An error if the format string is invalid or if replacement fields
            don't match the provided argument types.
        """
        comptime FormatOrigin = ImmutOrigin(format.origin)
        comptime EntryType = _FormatCurlyEntry[FormatOrigin]

        var manual_indexing_count = 0
        var automatic_indexing_count = 0
        var raised_manual_index = Optional[Int](None)
        var raised_automatic_index = Optional[Int](None)
        var raised_kwarg_field = Optional[StringSlice[FormatOrigin]](None)
        comptime n_args = Variadic.size(Ts)
        comptime `}` = UInt8(ord("}"))
        comptime `{` = UInt8(ord("{"))
        comptime l_err = "there is a single curly { left unclosed or unescaped"
        comptime r_err = "there is a single curly } left unclosed or unescaped"

        var entries = List[EntryType]()
        var start = Optional[Int](None)
        var skip_next = False
        var fmt_ptr = format.unsafe_ptr()
        var fmt_len = format.byte_length()
        var total_estimated_entry_byte_width = 0

        for i in range(fmt_len):
            if skip_next:
                skip_next = False
                continue
            if fmt_ptr[i] == `{`:
                if not start:
                    start = i
                    continue
                if i - start.value() != 1:
                    raise Error(l_err)
                # python escapes double curlies
                entries.append(EntryType(start.value(), i, field=False))
                start = None
                continue
            elif fmt_ptr[i] == `}`:
                if not start:
                    # python escapes double curlies
                    if (i + 1) < fmt_len and fmt_ptr[i + 1] == `}`:
                        entries.append(EntryType(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                    # if it is not an escaped one, it is an error
                    raise Error(r_err)

                var start_value = start.value()
                var current_entry = EntryType(start_value, i, field=NoneType())

                if i - start_value != 1:
                    if current_entry._handle_field_and_break(
                        format,
                        n_args,
                        i,
                        start_value,
                        automatic_indexing_count,
                        raised_automatic_index,
                        manual_indexing_count,
                        raised_manual_index,
                        raised_kwarg_field,
                        total_estimated_entry_byte_width,
                    ):
                        break
                else:  # automatic indexing
                    if automatic_indexing_count >= n_args:
                        raised_automatic_index = automatic_indexing_count
                        break
                    automatic_indexing_count += 1
                    total_estimated_entry_byte_width += 8  # guessing
                entries.append(current_entry^)
                start = None

        if raised_automatic_index:
            raise Error("Automatic indexing require more args in *args")
        elif raised_kwarg_field:
            var val = raised_kwarg_field.value()
            raise Error("Index ", val, " not in kwargs")
        elif manual_indexing_count and automatic_indexing_count:
            raise Error("Cannot both use manual and automatic indexing")
        elif raised_manual_index:
            var val = raised_manual_index.value()
            raise Error("Index ", val, " not in *args")
        elif start:
            raise Error(l_err)
        return {entries^, total_estimated_entry_byte_width, format}


# NOTE(#3765): an interesting idea would be to allow custom start and end
# characters for formatting (passed as parameters to Formatter), this would be
# useful for people developing custom templating engines as it would allow
# determining e.g. `<mojo` [...] `>` [...] `</mojo>` html tags.
# And going a step further it might even be worth it adding custom format
# specification start character, and custom format specs themselves (by defining
# a trait that all format specifications conform to)
struct _FormatCurlyEntry[
    origin: ImmutOrigin, spec_type: _FormatSpecType = _DefaultFormatSpec
](ImplicitlyCopyable):
    """The struct that handles string formatting by curly braces entries.
    This is internal for the types: `StringSlice` compatible types.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of a closing brace around a substitution field."""
    var format_spec: Self.spec_type
    """The format specifier."""
    comptime _FieldVariantType = Variant[
        StringSlice[Self.origin], Int, NoneType, Bool
    ]
    """Purpose of the `Variant` `Self.field`:

    - `Int` for manual indexing: (value field contains `0`).
    - `NoneType` for automatic indexing: (value field contains `None`).
    - `StringSlice` for **kwargs indexing: (value field contains `foo`).
    - `Bool` for escaped curlies: (value field contains False for `{` or True
        for `}`).
    """
    var field: Self._FieldVariantType
    """Store the substitution field. See `Self._FieldVariantType` docstrings for
    more details."""

    fn __init__(
        out self,
        first_curly: Int,
        last_curly: Int,
        field: Self._FieldVariantType,
        format_spec: Self.spec_type = {},
    ):
        """Construct a format entry.

        Args:
            first_curly: The index of an opening brace around a substitution
                field.
            last_curly: The index of a closing brace around a substitution
                field.
            field: Store the substitution field.
            format_spec: The format specifier.
        """
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.format_spec = format_spec

    @always_inline
    fn is_escaped_brace(ref self) -> Bool:
        """Whether the field is escaped_brace.

        Returns:
            The result.
        """
        return self.field.isa[Bool]()

    @always_inline
    fn is_kwargs_field(ref self) -> Bool:
        """Whether the field is kwargs_field.

        Returns:
            The result.
        """
        return self.field.isa[String]()

    @always_inline
    fn is_automatic_indexing(ref self) -> Bool:
        """Whether the field is automatic_indexing.

        Returns:
            The result.
        """
        return self.field.isa[NoneType]()

    @always_inline
    fn is_manual_indexing(ref self) -> Bool:
        """Whether the field is manual_indexing.

        Returns:
            The result.
        """
        return self.field.isa[Int]()

    fn _handle_field_and_break(
        mut self,
        fmt_src: StringSlice[Self.origin],
        len_pos_args: Int,
        i: Int,
        start_value: Int,
        mut automatic_indexing_count: Int,
        mut raised_automatic_index: Optional[Int],
        mut manual_indexing_count: Int,
        mut raised_manual_index: Optional[Int],
        mut raised_kwarg_field: Optional[StringSlice[Self.origin]],
        mut total_estimated_entry_byte_width: Int,
    ) raises -> Bool:
        @always_inline("nodebug")
        fn _build_slice(
            p: UnsafePointer[mut=False, UInt8], start: Int, end: Int
        ) -> StringSlice[p.origin]:
            return StringSlice(ptr=p + start, length=end - start)

        var field = _build_slice(fmt_src.unsafe_ptr(), start_value + 1, i)
        # FIXME: We shouldn't hardcode the potential format spec characters,
        # the implementation should go per element here and stop when a
        # character that doesn't match {`\w*`, `\d*`, `\[`, `\]`, `.`} is
        # encountered, and then parse the rest using the provided spec

        var fmt_start = max(field.find("!"), field.find(":"))
        var pre_fmt_field = field[:fmt_start] if fmt_start != -1 else field
        var break_from_loop = False
        if pre_fmt_field.byte_length() == 0:
            # an empty field, so it's automatic indexing
            if automatic_indexing_count >= len_pos_args:
                raised_automatic_index = automatic_indexing_count
                break_from_loop = True
            automatic_indexing_count += 1
        else:
            # TODO: add support for "My name is {0.name}".format(Person(name="Fred"))
            # TODO: add support for "My name is {person.name}".format(person=Person(name="Fred"))
            # NOTE: use reflection to access the fields, but be mindful of
            # nested accesses like
            # "Some: {0.who.name]}".format(Someone(who=Person(name="Fred")))

            # TODO: add support for "My name is {0[name]}".format({"name": "Fred"})
            # TODO: add support for "My name is {person[name]}".format(person={"name": "Fred"})
            # NOTE: This will require an Indexable parametric trait that
            # we'd have to check conformance to for the indexable type. When
            # it's a digit then `Indexer`, otherwise it has to be able
            # to be indexed by a `String` or a `StringSlice`.
            try:
                # field is a number for manual indexing:
                var number = Int(pre_fmt_field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    break_from_loop = True
                manual_indexing_count += 1
            except e:

                @parameter
                fn check_string() -> Bool:
                    return "not convertible to integer" in String(e)

                debug_assert[check_string]("Not the expected error from atol")
                # field is a keyword for **kwargs:
                self.field = pre_fmt_field
                raised_kwarg_field = pre_fmt_field
                break_from_loop = True

        self.format_spec = {
            field[fmt_start:]
        } if fmt_start != -1 else Self.spec_type()
        return break_from_loop

    fn _format_entry[
        len_pos_args: Int
    ](self, mut writer: Some[Writer], args: _FormatArgs, mut auto_idx: Int):
        comptime r_value = UInt8(ord("r"))
        comptime s_value = UInt8(ord("s"))

        fn _format(idx: Int) unified {read self, read args, mut writer}:
            comptime for i in range(len_pos_args):
                if i == idx:
                    var flag = self.format_spec.get_conversion_flag()
                    ref arg = trait_downcast[Writable](args[i])
                    if flag == s_value:
                        arg.write_to(writer)
                    elif flag == r_value:
                        arg.write_repr_to(writer)

        if self.is_escaped_brace():
            writer.write("}" if self.field[Bool] else "{")
        elif self.is_manual_indexing():
            _format(self.field[Int])
        elif self.is_automatic_indexing():
            _format(auto_idx)
            auto_idx += 1


# ===-----------------------------------------------------------------------===#
# Format Specification
# ===-----------------------------------------------------------------------===#


trait _FormatSpecType(Defaultable, ImplicitlyCopyable):
    comptime amnt_conversion_flags: Int = 2
    comptime valid_conversion_flags: InlineArray[
        UInt8, Self.amnt_conversion_flags
    ] = ([Byte(ord("s")), Byte(ord("r"))])

    fn __init__(out self, fmt_str: StringSlice) raises:
        """Parses the format spec from a string that should comply with the
        spec.

        Args:
            fmt_str: The StringSlice with the format spec.
        """
        ...

    # NOTE: this function is temporary until we integrate the API into Writable
    fn get_conversion_flag(self) -> UInt8:
        """Get the current conversion flag.

        Returns:
            The current conversion flag.
        """
        ...


struct _DefaultFormatSpec(TrivialRegisterType, _FormatSpecType):
    """A Mojo Format Specification, inspired in [Python's formatspec](
    https://docs.python.org/3/library/string.html#formatspec).

    #### The order of elements that this implementation expects is the \
    following (brackets signify optionality):
    - `[conversion_flag][:[fill_codepoint][alignment][grapheme_width]]`.
    """

    var conversion_flag: UInt8
    """The conversion flag: {Byte(ord("s")), Byte(ord("r"))}"""
    var fill: Codepoint
    """If a valid align value is specified, the fill codepoint is used. Defaults
    to a space."""
    var align: _Alignment
    """The alignment options."""
    var grapheme_width: UInt16
    """A decimal integer defining the minimum total field grapheme width,
    including any prefixes, separators, and other formatting characters. If not
    specified, then the field width will be determined by the content. When no
    explicit alignment is given, preceding the width field by a zero ('0')
    character enables sign-aware zero-padding for numeric types. This is
    equivalent to a fill character of '0' with an alignment type of '='.
    """

    fn __init__(out self):
        """Construct the default."""
        self = {conversion_flag = Byte(ord("s"))}

    fn __init__(out self, fmt_str: StringSlice) raises:
        """Parses the format spec from a string that should comply with the
        spec.

        Args:
            fmt_str: The StringSlice with the format spec.
        """
        # TODO: implement this properly with tests
        var data = fmt_str.as_bytes()
        if fmt_str.byte_length() == 0:
            return {}
        elif data[0] == Byte(ord("!")):
            if fmt_str.byte_length() < 2:
                raise Error("Empty conversion flag.")
            elif fmt_str.byte_length() > 2 and data[2] != Byte(ord(":")):
                raise Error(
                    'Conversion flag "', fmt_str[1:], '" not recognized.'
                )
            var flag = data[1]
            if not flag in materialize[Self.valid_conversion_flags]():
                raise Error(
                    'Conversion flag "', Codepoint(flag), '" not recognized.'
                )
            return {conversion_flag = flag}

        raise Error("Not implemented")

    @always_inline
    fn __init__(
        out self,
        *,
        conversion_flag: UInt8,
        fill: Codepoint = Codepoint.ord(" "),
        align: _Alignment = _Alignment.LEFT,
        grapheme_width: UInt16 = 0,
    ):
        """Construct a `BaseFormatSpec`.

        Args:
            conversion_flag: The conversion flag.
            fill: The codepoint to fill with, defaults to " ".
            align: The alignment options.
            grapheme_width: The minimum grapheme width of the result.
        """
        self.conversion_flag = conversion_flag
        self.fill = fill
        self.align = align
        self.grapheme_width = grapheme_width

    fn get_conversion_flag(self) -> UInt8:
        """Get the current conversion flag.

        Returns:
            The current conversion flag.
        """
        return self.conversion_flag


@fieldwise_init
struct _Alignment(Equatable, TrivialRegisterType):
    comptime LEFT = Self(Byte(ord("<")))
    """Forces the field to be left-aligned within the available space
    (this is the default for most objects)."""
    comptime RIGHT = Self(Byte(ord(">")))
    """Forces the field to be right-aligned within the available space
    (this is the default for numbers)."""
    comptime EQUAL = Self(Byte(ord("=")))
    """Forces the padding to be placed after the sign (if any) but before
    the digits. This is used for printing fields in the form `+000000120`. This
    alignment option is only valid for numeric types. It becomes the default
    for numbers when `0` immediately precedes the field width.
    """
    comptime CENTERED = Self(Byte(ord("^")))
    """Forces the field to be centered within the available space."""

    var _value: UInt8


struct _NumericFormatSpec(TrivialRegisterType, _FormatSpecType):
    """A Mojo Format Specification, inspired in [Python's formatspec](
    https://docs.python.org/3/library/string.html#formatspec).

    #### The order of elements that this implementation expects is the \
    following (brackets signify optionality):
    - `[!conversion_flag][:[fill_codepoint][alignment][sign][grapheme_width][numeric_section]]`
    - `numeric_section = [[float_options][thousand_separator][float_form][integer_form]]`
    - `float_options = [["z"][.precision]]`
    - `float_form = [{"e", "E", "f", "F", "g", "G"}]`
    - `integer_form = [{"d", "b", "x", "X", "o", "c"}]`
    """

    var base: _DefaultFormatSpec
    """The base format spec for the number."""
    var sign: _SignOptions
    """The sign options."""
    var thousand_separator: UInt8
    """The ASCII character to use as a separator."""
    var float_options: _FloatOptions
    """Options for dealing with floating point numbers."""
    var int_options: _IntOptions
    """Options for dealing with integers."""

    fn __init__(out self):
        """Construct the default."""
        self = {{conversion_flag = Byte(ord("s")), align = _Alignment.RIGHT}}

    fn __init__(out self, fmt_str: StringSlice) raises:
        """Parses the format spec from a string that should comply with the
        spec.

        Args:
            fmt_str: The StringSlice with the format spec.
        """
        # TODO: implement this properly with tests
        var data = fmt_str.as_bytes()
        if fmt_str.byte_length() == 0:
            return {}
        elif data[0] == Byte(ord("!")):
            if fmt_str.byte_length() < 2:
                raise Error("Empty conversion flag.")
            elif fmt_str.byte_length() > 2 and data[2] != Byte(ord(":")):
                raise Error(
                    'Conversion flag "', fmt_str[1:], '" not recognized.'
                )
            var flag = data[1]
            if flag not in materialize[Self.valid_conversion_flags]():
                raise Error(
                    'Conversion flag "', Codepoint(flag), '" not recognized.'
                )
            return {{conversion_flag = flag}}
        if data[0] == Byte(ord(":")) and fmt_str.byte_length() == 1:
            raise Error("Empty format after ':'")

        raise Error("Not implemented")

    fn __init__(
        out self,
        base: _DefaultFormatSpec,
        *,
        sign: _SignOptions = _SignOptions.ONLY_NEGATIVE,
        thousand_separator: UInt8 = Byte(ord("")),
        float_options: _FloatOptions = {},
        int_options: _IntOptions = {},
    ):
        """Construct a `BaseFormatSpec`.

        Args:
            base: The base format spec for the number.
            sign: The sign options.
            thousand_separator: The ASCII character to use as a separator.
            float_options: Options for dealing with floating point numbers.
            int_options: Options for dealing with integers.
        """
        self.base = base
        self.sign = sign
        self.thousand_separator = thousand_separator
        self.float_options = float_options
        self.int_options = int_options

    fn get_conversion_flag(self) -> UInt8:
        """Get the current conversion flag.

        Returns:
            The current conversion flag.
        """
        return self.base.conversion_flag


@fieldwise_init
struct _SignOptions(Equatable, TrivialRegisterType):
    comptime BOTH = Self(Byte(ord("+")))
    """Indicates that a sign should be used for both positive as well as
    negative numbers."""
    comptime ONLY_NEGATIVE = Self(Byte(ord("-")))
    """Indicates that a sign should be used only for negative numbers (this
    is the default behavior)."""
    comptime SPACE = Self(Byte(ord(" ")))
    """Indicates that a leading space should be used on positive numbers,
    and a minus sign on negative numbers."""

    var _value: UInt8


@fieldwise_init
struct _IntOptions(Defaultable, Equatable, TrivialRegisterType):
    """Options for dealing with integers."""

    var form: _IntForm
    """The form the integer should take."""

    fn __init__(out self):
        self.form = _IntForm.DECIMAL


struct _FloatOptions(Defaultable, Equatable, TrivialRegisterType):
    """Options for dealing with floating point numbers."""

    var coerce_z: Bool
    """The 'z' option coerces negative zero floating-point values to positive
    zero after rounding to the format precision. This option is only valid for
    floating-point presentation types."""
    var precision: UInt64
    """The precision is a decimal integer indicating how many digits should be
    displayed after the decimal point for presentation types 'f' and 'F', or
    before and after the decimal point for presentation types 'g' or 'G'. For
    string presentation types the field indicates the maximum field size - in
    other words, how many characters will be used from the field content. The
    precision is not allowed for integer presentation types.
    """
    var form: _FloatForm
    """The form the float should take."""

    fn __init__(out self):
        self.precision = UInt64.MAX
        self.coerce_z = False
        self.form = _FloatForm.LOWER_GENERAL_FORMAT


@fieldwise_init
struct _IntForm(Equatable, TrivialRegisterType):
    """The numeric form of the value to use."""

    comptime BINARY = Self(Byte(ord("b")))
    """Outputs the number in base 2."""
    comptime CHARACTER = Self(Byte(ord("c")))
    """Converts the integer to the corresponding unicode character before
    printing."""
    comptime DECIMAL = Self(Byte(ord("d")))
    """Outputs the number in base 10."""
    comptime OCTAL = Self(Byte(ord("o")))
    """Octal format. Outputs the number in base 8."""
    comptime LOWER_HEX = Self(Byte(ord("x")))
    """Hex format. Outputs the number in base 16, using lower-case letters
    for the digits above 9."""
    comptime UPPER_HEX = Self(Byte(ord("X")))
    """Hex format. Outputs the number in base 16, using upper-case letters
    for the digits above 9."""

    var _value: UInt8


@fieldwise_init
struct _FloatForm(Equatable, TrivialRegisterType):
    """The numeric form of the value to use."""

    comptime LOWER_SCIENTIFIC = Self(Byte(ord("e")))
    """For a given precision p, formats the number in
    scientific notation with the letter `e` separating the coefficient from the
    exponent. The coefficient has one digit before and p digits after the
    decimal point, for a total of p + 1 significant digits. With no precision
    given, uses a precision of 6 digits after the decimal point for float, and
    shows all coefficient digits for Decimal. If no digits follow the decimal
    point, the decimal point is also removed unless the # option is used."""
    comptime UPPER_SCIENTIFIC = Self(Byte(ord("E")))
    """Same as 'e' except it uses an upper case `E` as the separator
    character."""
    comptime LOWER_FIXED_POINT = Self(Byte(ord("f")))
    """For a given precision p, formats the number as
    a decimal number with exactly p digits following the decimal point. With no
    precision given, uses a precision of 6 digits after the decimal point for
    float, and uses a precision large enough to show all coefficient digits for
    Decimal. If no digits follow the decimal point, the decimal point is also
    removed unless the '#' option is used."""
    comptime UPPER_FIXED_POINT = Self(Byte(ord("F")))
    """Same as 'f', but converts nan to NAN and inf to INF."""
    comptime LOWER_GENERAL_FORMAT = Self(Byte(ord("g")))
    """For a given precision p >= 1, this rounds the number
    to p significant digits and then formats the result in either fixed-point
    format or in scientific notation, depending on its magnitude. A precision
    of 0 is treated as equivalent to a precision of 1.
    The precise rules are as follows: suppose that the result formatted with
    presentation type 'e' and precision p-1 would have exponent exp. Then, if
    m <= exp < p, where m is -4 for floats and -6 for Decimals, the number is
    formatted with presentation type 'f' and precision p-1-exp. Otherwise, the
    number is formatted with presentation type 'e' and precision p-1. In both
    cases insignificant trailing zeros are removed from the significand, and
    the decimal point is also removed if there are no remaining digits
    following it, unless the '#' option is used.
    With no precision given, uses a precision of 6 significant digits for
    float. For Decimal, the coefficient of the result is formed from the
    coefficient digits of the value; scientific notation is used for values
    smaller than 1e-6 in absolute value and values where the place value of the
    least significant digit is larger than 1, and fixed-point notation is used
    otherwise.
    Positive and negative infinity, positive and negative zero, and nans, are
    formatted as inf, -inf, 0, -0 and nan respectively, regardless of the
    precision.
    """
    comptime UPPER_GENERAL_FORMAT = Self(Byte(ord("G")))
    """Same as 'g' except switches to 'E' if the number gets
    too large. The representations of infinity and NaN are uppercased, too."""

    var _value: UInt8


# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#
