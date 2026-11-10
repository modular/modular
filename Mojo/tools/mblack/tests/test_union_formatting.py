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

import pytest

from tests.util import assert_mojo_format


def test_union_is_stable():
    source = """\
__union Tok(Copyable, Writable):
    \"\"\"A lexer token.\"\"\"

    case eof()
    \"\"\"The end of the input.\"\"\"
    case word(String)
    case span(Int, Int)

    def is_eof(self) -> Bool:
        match self:
            case .eof():
                return True
            case _:
                return False

    @staticmethod
    def default() -> Self:
        return Self.eof()

    comptime Name = "Tok"
"""
    assert_mojo_format(source, source)


@pytest.mark.parametrize("kw", ["__enum", "__union"])
def test_union_body_formats_like_enum_body(kw):
    source = f"""\
{kw} Tok(Writable,Copyable):
    \"\"\"A lexer token.\"\"\"
    case eof ()
    \"\"\"The end of the input.\"\"\"
    case word( String )


    case span(Int,Int)
    case long_payload_case_name(List[Int], Dict[String, Int], Optional[String], Int)
    def is_eof(self) -> Bool:
        return False
    comptime Name = "Tok"
"""
    expected = f"""\
{kw} Tok(Copyable, Writable):
    \"\"\"A lexer token.\"\"\"

    case eof()
    \"\"\"The end of the input.\"\"\"
    case word(String)

    case span(Int, Int)
    case long_payload_case_name(
        List[Int], Dict[String, Int], Optional[String], Int
    )

    def is_eof(self) -> Bool:
        return False

    comptime Name = "Tok"
"""
    assert_mojo_format(source, expected)


def test_union_one_line_body_and_semicolons():
    source = """\
__union Unit: case only()
__union Pair:
    case a(); case b(Int)
"""
    expected = """\
__union Unit:
    case only()


__union Pair:
    case a()
    case b(Int)
"""
    assert_mojo_format(source, expected)


def test_union_case_without_parens_is_kept():
    """The compiler, not the formatter, judges which cases need parentheses."""
    source = """\
__union Maybe:
    case none
    case some(Int)
"""
    assert_mojo_format(source, source)


@pytest.mark.parametrize("kw", ["struct", "__union"])
def test_union_header_splits_like_struct(kw):
    """`__union` is a column wider than `struct`; no line is near the limit."""
    source = (
        f"{kw} Result[T: Copyable & Deinitable, E: Copyable & Deinitable]"
        "(Movable, Copyable where conforms_to(T, Copyable)):\n"
        "    pass\n"
        f"{kw} Wide[T: AnyType](Movable) where conforms_to(T, Copyable)"
        " and conforms_to(T, Movable) and conforms_to(T, Writable):\n"
        "    pass\n"
    )
    expected = (
        f"{kw} Result[T: Copyable & Deinitable, E: Copyable & Deinitable](\n"
        "    Copyable where conforms_to(T, Copyable), Movable\n"
        "):\n"
        "    pass\n"
        "\n"
        "\n"
        f"{kw} Wide[T: AnyType](Movable) where (\n"
        "    conforms_to(T, Copyable)\n"
        "    and conforms_to(T, Movable)\n"
        "    and conforms_to(T, Writable)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", ["struct", "__union"])
def test_union_name_may_be_a_keyword_like_struct(kw):
    names = ["where", "raises", "capturing", "escaping"]
    source = "".join(
        f"{kw} {n}(Writable,Copyable):\n    pass\n" for n in names
    )
    expected = "\n\n".join(
        f"{kw} {n}(Copyable, Writable):\n    pass\n" for n in names
    )
    assert_mojo_format(source, expected)


def test_union_after_simple_statement():
    source = """\
from std.os import abort
__union Tok:
    case eof()
"""
    expected = """\
from std.os import abort


__union Tok:
    case eof()
"""
    assert_mojo_format(source, expected)
