# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_underscores_in_integer():
    """Underscores in integer literals."""
    source = (
        "def main():\n"
        "    print(1_000)\n"
        "    print(1__000)\n"
        "    print(1___000)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_float():
    """Underscores in float literals."""
    source = (
        "def main():\n"
        "    print(1.5_0)\n"
        "    print(0.5_0)\n"
        "    print(5_0.0)\n"
        "    print(12.3_1)\n"
        "    print(12.3__1)\n"
        "    print(12__3.1)\n"
        "    print(0.5__0)\n"
        "    print(0.5___000)\n"
        "    print(0.1__2__3)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_exponent_float():
    """Underscores in exponent float literals."""
    source = (
        "def main():\n"
        "    print(1e1_0)\n"
        "    print(1e1__0)\n"
        "    print(1__0e5)\n"
        "    print(1e-2__3)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_binary():
    """Underscores in binary literals."""
    source = (
        "def main():\n"
        "    print(0b1_0)\n"
        "    print(0b1__0)\n"
        "    print(0b_1)\n"
        "    print(0b__1)\n"
        "    print(0b_1__0)\n"
        "    print(0b1___0___1)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_hex():
    """Underscores in hex literals."""
    source = (
        "def main():\n"
        "    print(0xFF_FF)\n"
        "    print(0xFF__FF)\n"
        "    print(0x_FF)\n"
        "    print(0x__FF)\n"
        "    print(0x_FF__FF)\n"
        "    print(0x_DE__AD__BE__EF)\n"
    )
    assert_mojo_format(source, source)


def test_underscores_in_octal():
    """Underscores in octal literals."""
    source = (
        "def main():\n"
        "    print(0o7_7)\n"
        "    print(0o7__7)\n"
        "    print(0o_7)\n"
        "    print(0o__7)\n"
        "    print(0o_7__7)\n"
    )
    assert_mojo_format(source, source)


def test_baseline_forms_roundtrip():
    """Canonical numeric literals without underscores round-trip unchanged."""
    source = (
        "def main():\n"
        "    print(0.5)\n"
        "    print(5e2)\n"
        "    print(5.0e2)\n"
        "    print(5.0e-2)\n"
        "    print(0b10)\n"
        "    print(0x1F)\n"
        "    print(0o77)\n"
    )
    assert_mojo_format(source, source)


def test_float_dot_canonicalization():
    """Leading-dot floats get a `0` prefix; trailing-dot floats get a `0` suffix."""
    source = (
        "def main():\n"
        # Leading dot gets a `0` prefix.
        "    print(.5)\n"
        "    print(.5_0)\n"
        "    print(.5__0)\n"
        "    print(.5___000)\n"
        "    print(.1__2__3)\n"
        # Trailing dot gets a `0` suffix.
        "    print(5.)\n"
        "    print(5_0.)\n"
    )
    expected = (
        "def main():\n"
        "    print(0.5)\n"
        "    print(0.5_0)\n"
        "    print(0.5__0)\n"
        "    print(0.5___000)\n"
        "    print(0.1__2__3)\n"
        "    print(5.0)\n"
        "    print(5_0.0)\n"
    )
    assert_mojo_format(source, expected)


def test_exponent_sign_normalization():
    """Explicit `+` in exponents is removed; `-` is preserved. Combines with dot
    canonicalization."""
    source = (
        "def main():\n"
        # Explicit `+` is removed.
        "    print(1e+2__3)\n"
        "    print(1.0e+5)\n"
        # `-` is preserved, including under leading-/trailing-dot canonicalization.
        "    print(.1e-3)\n"
        "    print(5_0.e-2)\n"
        # Combines with leading-/trailing-dot canonicalization.
        "    print(.5e2)\n"
        "    print(.1__2e-3__4)\n"
        "    print(5_0.e+2)\n"
    )
    expected = (
        "def main():\n"
        "    print(1e2__3)\n"
        "    print(1.0e5)\n"
        "    print(0.1e-3)\n"
        "    print(5_0.0e-2)\n"
        "    print(0.5e2)\n"
        "    print(0.1__2e-3__4)\n"
        "    print(5_0.0e2)\n"
    )
    assert_mojo_format(source, expected)


def test_numeric_literal_case_normalization():
    """Uppercase exponent letters and base prefixes lowercase; hex digits uppercase."""
    source = (
        "def main():\n"
        # Uppercase exponent letter.
        "    print(1E2)\n"
        "    print(1E+2)\n"
        "    print(1E-2__3)\n"
        # Uppercase base prefix.
        "    print(0X1F)\n"
        "    print(0B10)\n"
        "    print(0O77)\n"
        # Hex digits are normalized to uppercase regardless of input case.
        "    print(0xaf_af)\n"
        "    print(0xAF_AF)\n"
        "    print(0X_de__ad)\n"
    )
    expected = (
        "def main():\n"
        "    print(1e2)\n"
        "    print(1e2)\n"
        "    print(1e-2__3)\n"
        "    print(0x1F)\n"
        "    print(0b10)\n"
        "    print(0o77)\n"
        "    print(0xAF_AF)\n"
        "    print(0xAF_AF)\n"
        "    print(0x_DE__AD)\n"
    )
    assert_mojo_format(source, expected)
