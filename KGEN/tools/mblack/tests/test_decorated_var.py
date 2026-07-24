# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_decorated_var():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var  x:  Int\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_decorated_var_multiple_fields():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    assert_mojo_format(source, expected)
