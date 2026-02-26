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
from collections.string.format import _FormatUtils
import format._utils as fmt


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

    fn write_to(self, mut writer: Some[Writer]):
        """Write the formatted string to a writer.

        This method implements the `Writable` trait by formatting the TString's
        template with its interpolated values and writing the result to the
        provided writer. The format string is compiled at compile-time for
        optimal performance.

        Args:
            writer: The writer to output the formatted string to.
        """
        _FormatUtils.format_to_comptime[Self.format_string](
            writer, self._values
        )

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
