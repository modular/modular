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


def test_enum_is_stable():
    source = """\
__enum Token(Copyable, Writable):
    \"\"\"A lexer token.\"\"\"

    case eof
    case identifier(String)
    case span(Int, Int)

    def is_eof(self) -> Bool:
        __match self:
            case .eof:
                return True
            case _:
                return False

    @staticmethod
    def default() -> Self:
        return Self.eof

    comptime Name = "Token"
"""
    assert_mojo_format(source, source)


def test_enum_body_formats_like_struct_body():
    """Cases space like fields; methods and members get one blank line."""
    source = """\
__enum Token(Writable, Copyable):
    \"\"\"A lexer token.\"\"\"
    case eof
    case identifier( String )


    case span(Int,Int)
    def is_eof(self) -> Bool:
        return False
    comptime Name = "Token"
"""
    expected = """\
__enum Token(Copyable, Writable):
    \"\"\"A lexer token.\"\"\"

    case eof
    case identifier(String)

    case span(Int, Int)

    def is_eof(self) -> Bool:
        return False

    comptime Name = "Token"
"""
    assert_mojo_format(source, expected)


def test_enum_case_doc_strings():
    """A doc string stays on the line after its case, as after a field."""
    source = """\
__enum Color:
    \"\"\"A color.\"\"\"
    case red
    \"\"\"Pure red.\"\"\"
    case green

    case rgb(Int, Int, Int)
    \"\"\"
    A color by its components.
    \"\"\"
"""
    expected = """\
__enum Color:
    \"\"\"A color.\"\"\"

    case red
    \"\"\"Pure red.\"\"\"
    case green

    case rgb(Int, Int, Int)
    \"\"\"
    A color by its components.
    \"\"\"
"""
    assert_mojo_format(source, expected)


def test_enum_one_line_body_and_semicolons():
    source = """\
__enum Unit: case only
__enum Pair:
    case a; case b(Int)
"""
    expected = """\
__enum Unit:
    case only


__enum Pair:
    case a
    case b(Int)
"""
    assert_mojo_format(source, expected)


def test_enum_case_payload_splits_like_a_call():
    source = """\
__enum Payloads:
    case long_payload_case_name(List[Int], Dict[String, Int], Optional[String], Int)
    case trailing_comma(Int, Bool,)
    case multi_line(
        Int,
        Bool
    )
    case `def`
"""
    expected = """\
__enum Payloads:
    case long_payload_case_name(
        List[Int], Dict[String, Int], Optional[String], Int
    )
    case trailing_comma(
        Int,
        Bool,
    )
    case multi_line(Int, Bool)
    case `def`
"""
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", ["struct", "__enum"])
def test_enum_header_splits_like_struct(kw):
    """An `__enum` header is split exactly as the same `struct` header."""
    source = (
        f"{kw} Result[T: Copyable & Deinitable, E: Copyable & Deinitable]"
        "(Movable, Copyable where conforms_to(T, Copyable)):\n"
        "    pass\n"
        f"{kw} Wide[T: AnyType](Movable) where conforms_to(T, Copyable)"
        " and conforms_to(T, Movable):\n"
        "    pass\n"
    )
    expected = (
        f"{kw} Result[T: Copyable & Deinitable, E: Copyable & Deinitable](\n"
        "    Copyable where conforms_to(T, Copyable), Movable\n"
        "):\n"
        "    pass\n"
        "\n"
        "\n"
        f"{kw} Wide[T: AnyType](Movable) where conforms_to(T, Copyable)"
        " and conforms_to(\n"
        "    T, Movable\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", ["struct", "__enum"])
def test_enum_name_may_be_a_keyword_like_struct(kw):
    """After `__enum`, as after `struct`, these keywords are a name."""
    names = ["where", "raises", "capturing", "escaping"]
    source = "".join(
        f"{kw} {n}(Writable,Copyable):\n    pass\n" for n in names
    )
    expected = "\n\n".join(
        f"{kw} {n}(Copyable, Writable):\n    pass\n" for n in names
    )
    assert_mojo_format(source, expected)


def test_enum_after_simple_statement():
    source = """\
from std.os import abort
__enum Token:
    case eof
"""
    expected = """\
from std.os import abort


__enum Token:
    case eof
"""
    assert_mojo_format(source, expected)


def test_case_as_name_and_in_match_is_unchanged():
    """`case` as a name and in `__match` is unaffected by enum cases."""
    source = """\
__enum E:
    case a

    def case(self) -> Int:
        var case = 1
        case += 1
        case.bit_count()
        __match self:
            case .a:
                return case
"""
    assert_mojo_format(source, source)
