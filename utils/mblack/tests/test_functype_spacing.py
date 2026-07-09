# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_functype_no_space_before_parens():
    """No space between metaparams and parens in function types."""
    source = "def takes_bar[bar: def[x: Int]()](): pass"
    expected = "def takes_bar[bar: def[x: Int]()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_space_before_parens_with_args():
    """No space between metaparams and parens in function types with args."""
    source = "def takes_bar[bar: def[x: Int](y: Int)](): pass"
    expected = "def takes_bar[bar: def[x: Int](y: Int)]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_metaparams():
    """Function types without metaparams should also not have extra space."""
    source = "def takes_bar[bar: def()](): pass"
    expected = "def takes_bar[bar: def()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_with_return_type():
    """Function types with return types should format correctly."""
    source = "def takes_bar[bar: def[x: Int]() -> Int](): pass"
    expected = "def takes_bar[bar: def[x: Int]() -> Int]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_named_effect_no_space_before_parens():
    """No space between effect name and parens (e.g., abi("C"))."""
    source = 'def foo() abi ("C"):\n    pass\n'
    expected = 'def foo() abi("C"):\n    pass\n'
    assert_mojo_format(source, expected)


def test_named_effect_no_space_already_correct():
    """abi("C") without space should remain unchanged."""
    source = 'def foo() abi("C"):\n    pass\n'
    expected = 'def foo() abi("C"):\n    pass\n'
    assert_mojo_format(source, expected)


def test_named_effect_stays_inline_when_signature_wraps():
    """A named effect stays inline when a long function type must wrap.

    `abi("C")` is not a split point, so the wrap lands on the return-type
    subscript instead.
    """
    source = (
        "def main() raises:\n"
        "    var curl_version = lib.get_function[\n"
        '        def() thin abi("C") -> UnsafePointer[\n'
        "            c_char, ImmutOrigin(origin_of(result))\n"
        "        ]\n"
        '    ]("curl_version")\n'
    )
    assert_mojo_format(source, source)
