# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_global_comptime():
    source = (
        "comptime   b =  6\n"
    )
    expected = (
        "comptime b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_nested_comptime():
    source = (
        "struct Foo:\n"
        "    comptime  b =  6\n"
    )
    expected = (
        "struct Foo:\n"
        "    comptime b = 6\n"
    )
    assert_mojo_format(source, expected)
