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
var s = "{!r}".format(myComplicatedObject)
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


from builtin.variadics import Variadic
from compile import get_type_name
from utils import Variant

# ===-----------------------------------------------------------------------===#
# Formatter
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _PrecompiledEntries[origin: ImmutOrigin, //, *Ts: AnyType](Movable):
    var entries: List[_FormatCurlyEntry[Self.origin]]
    var size_hint: Int
    var format: StringSlice[Self.origin]


comptime _FormatArgs = VariadicPack[element_trait=Writable, ...]


struct _FormatUtils:
    # TODO: Have this return a `Result[_PrecompiledEntries, Error]`
    @staticmethod
    fn compile_entries[
        *Ts: Writable
    ](format: StringSlice) -> Variant[
        _PrecompiledEntries[origin = ImmutOrigin(format.origin), *Ts],
        Error,
    ]:
        """Precompile the entries using the given format string."""
        try:
            return Self._compile_entries[*Ts](format)
        except e:
            return e^

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
            debug_assert(offset < fmt_len, "offset >= format.byte_length()")
            writer.write(_build_slice(ptr, offset, e.first_curly))
            e._format_entry[len_pos_args](writer, args, auto_arg_index)
            offset = e.last_curly + 1

        writer.write(_build_slice(ptr, offset, fmt_len))

    @staticmethod
    fn format(
        format: StringSlice, args: VariadicPack[element_trait=Writable, ...]
    ) raises -> String:
        """Format the arguments using the given format string."""
        comptime PackType = type_of(args)
        var compiled = Self._compile_entries[*PackType.element_types](format)

        var res = String(capacity=format.byte_length() + compiled.size_hint)
        Self.format_precompiled(writer=res, compiled=compiled, args=args)
        return res^

    @staticmethod
    fn _compile_entries[
        *Ts: Writable
    ](
        format: StringSlice,
    ) raises -> _PrecompiledEntries[
        origin = ImmutOrigin(format.origin), *Ts
    ]:
        """Returns a list of entries and its total estimated entry byte width.
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
                if not start and (i + 1) < fmt_len:
                    # python escapes double curlies
                    if fmt_ptr[i + 1] == `}`:
                        entries.append(EntryType(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                elif not start:  # if it is not an escaped one, it is an error
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
struct _FormatCurlyEntry[origin: ImmutOrigin](ImplicitlyCopyable):
    """The struct that handles string formatting by curly braces entries.
    This is internal for the types: `StringSlice` compatible types.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of a closing brace around a substitution field."""
    # TODO: ord("a") conversion flag not supported yet
    var conversion_flag: UInt8
    """The type of conversion for the entry: {ord("s"), ord("r")}."""
    # TODO: ord("a") conversion flag not supported yet
    comptime supported_conversion_flags = SIMD[DType.uint8, 2](
        UInt8(ord("s")), UInt8(ord("r"))
    )
    """Currently supported conversion flags: `__str__` and `__repr__`."""
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
        conversion_flag: UInt8 = 0,
    ):
        """Construct a format entry.

        Args:
            first_curly: The index of an opening brace around a substitution
                field.
            last_curly: The index of a closing brace around a substitution
                field.
            field: Store the substitution field.
            conversion_flag: The type of conversion for the entry.
        """
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.conversion_flag = conversion_flag

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
        var field_ptr = field.unsafe_ptr()
        var field_len = i - (start_value + 1)
        var exclamation_index = -1
        var idx = 0
        while idx < field_len:
            if field_ptr[idx] == UInt8(ord("!")):
                exclamation_index = idx
                break
            idx += 1
        var new_idx = exclamation_index + 1
        if exclamation_index != -1:
            if new_idx == field_len:
                raise Error("Empty conversion flag.")
            var conversion_flag = field_ptr[new_idx]
            if field_len - new_idx > 1 or (
                conversion_flag not in Self.supported_conversion_flags
            ):
                var f = _build_slice(field_ptr, new_idx, field_len)
                raise Error('Conversion flag "', f, '" not recognized.')
            self.conversion_flag = conversion_flag
            field = _build_slice(field_ptr, 0, exclamation_index)
        else:
            new_idx += 1

        # TODO(MSTDL-2243): Add format spec parsing

        if field.byte_length() == 0:
            # an empty field, so it's automatic indexing
            if automatic_indexing_count >= len_pos_args:
                raised_automatic_index = automatic_indexing_count
                return True
            automatic_indexing_count += 1
        else:
            try:
                # field is a number for manual indexing:
                # TODO: add support for "My name is {0.name}".format(Person(name="Fred"))
                # TODO: add support for "My name is {0[name]}".format({"name": "Fred"})
                var number = Int(field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    return True
                manual_indexing_count += 1
            except e:

                @parameter
                fn check_string() -> Bool:
                    return "not convertible to integer" in String(e)

                debug_assert[check_string]("Not the expected error from atol")
                # field is a keyword for **kwargs:
                # TODO: add support for "My name is {person.name}".format(person=Person(name="Fred"))
                # TODO: add support for "My name is {person[name]}".format(person={"name": "Fred"})
                var f = field
                self.field = f
                raised_kwarg_field = f
                return True
        return False

    fn _format_entry[
        len_pos_args: Int
    ](self, mut writer: Some[Writer], args: _FormatArgs, mut auto_idx: Int):
        # TODO(#3403 and/or #3252): this function should be able to use
        # Writer syntax when the type implements it, since it will give great
        # performance benefits. This also needs to be able to check if the given
        # args[i] conforms to the trait needed by the conversion_flag to avoid
        # needing to constraint that every type needs to conform to every trait.
        comptime r_value = UInt8(ord("r"))
        comptime s_value = UInt8(ord("s"))
        # alias a_value = UInt8(ord("a")) # TODO

        fn _format(idx: Int) unified {read self, read args, mut writer}:
            @parameter
            for i in range(len_pos_args):
                if i == idx:
                    var flag = self.conversion_flag
                    var empty = flag == 0

                    ref arg = trait_downcast[Writable](args[i])
                    if empty or flag == s_value:
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
