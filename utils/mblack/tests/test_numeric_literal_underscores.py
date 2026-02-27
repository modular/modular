# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_underscores_in_integer():
    """Underscores in integer literals."""
    source = (
        "fn main():\n"
        "    print(1_000)\n"
        "    print(1__000)\n"
        "    print(1___000)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_float():
    """Underscores in float literals."""
    source = (
        "fn main():\n"
        "    print(12.3_1)\n"
        "    print(12.3__1)\n"
        "    print(12__3.1)\n"
        "    print(.5__0)\n"
    )
    expected = (
        "fn main():\n"
        "    print(12.3_1)\n"
        "    print(12.3__1)\n"
        "    print(12__3.1)\n"
        "    print(0.5__0)\n"
    )
    assert_mojo_format(source, expected)


def test_underscores_in_exponent_float():
    """Underscores in exponent float literals."""
    source = (
        "fn main():\n"
        "    print(1e1_0)\n"
        "    print(1e1__0)\n"
        "    print(1__0e5)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_binary():
    """Underscores in binary literals."""
    source = (
        "fn main():\n"
        "    print(0b1_0)\n"
        "    print(0b1__0)\n"
        "    print(0b_1)\n"
        "    print(0b__1)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_hex():
    """Underscores in hex literals."""
    source = (
        "fn main():\n"
        "    print(0xFF_FF)\n"
        "    print(0xFF__FF)\n"
        "    print(0x_FF)\n"
        "    print(0x__FF)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_octal():
    """Underscores in octal literals."""
    source = (
        "fn main():\n"
        "    print(0o7_7)\n"
        "    print(0o7__7)\n"
        "    print(0o_7)\n"
        "    print(0o__7)\n"
    )
    assert_mojo_format(source, source)
