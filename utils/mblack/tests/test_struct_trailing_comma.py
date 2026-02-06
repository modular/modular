# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_struct_trailing_comma_params_with_conformances():
    """Trailing comma in struct params should expand when conformances are present."""
    source = (
        "struct MyStruct[A: Copyable, size: Int,](\n"
        "    Copyable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct MyStruct[\n"
        "    A: Copyable,\n"
        "    size: Int,\n"
        "](\n"
        "    Copyable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_struct_trailing_comma_params_and_conformances():
    """Trailing comma in both params and single-line conformances."""
    source = "struct MyStruct[A: Copyable, size: Int,](Copyable,):\n    pass\n"
    expected = (
        "struct MyStruct[\n"
        "    A: Copyable,\n"
        "    size: Int,\n"
        "](\n"
        "    Copyable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_struct_trailing_comma_params_without_conformances():
    """Trailing comma in struct params without conformances already works."""
    source = "struct MyStruct[A: Copyable, size: Int,]:\n    pass\n"
    expected = (
        "struct MyStruct[\n"
        "    A: Copyable,\n"
        "    size: Int,\n"
        "]:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
