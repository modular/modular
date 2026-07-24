# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_decorated_alias():
    source = (
        "struct Foo:\n"
        '    @deprecated("some message here")\n'
        "    alias  b =  6\n"
    )
    expected = (
        "struct Foo:\n"
        '    @deprecated("some message here")\n'
        "    alias b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_decorated_alias_with_args():
    source = (
        "struct Foo:\n"
        '    @deprecated(    "abc")\n'
        "    alias  b =  6\n"
    )
    expected = (
        "struct Foo:\n"
        '    @deprecated("abc")\n'
        "    alias b = 6\n"
    )
    assert_mojo_format(source, expected)
