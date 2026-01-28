# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_functype_no_space_before_parens():
    """No space between metaparams and parens in function types."""
    source = "fn takes_bar[bar: fn[x: Int]()](): pass"
    expected = "fn takes_bar[bar: fn[x: Int]()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_space_before_parens_with_args():
    """No space between metaparams and parens in function types with args."""
    source = "fn takes_bar[bar: fn[x: Int](y: Int)](): pass"
    expected = "fn takes_bar[bar: fn[x: Int](y: Int)]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_metaparams():
    """Function types without metaparams should also not have extra space."""
    source = "fn takes_bar[bar: fn()](): pass"
    expected = "fn takes_bar[bar: fn()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_with_return_type():
    """Function types with return types should format correctly."""
    source = "fn takes_bar[bar: fn[x: Int]() -> Int](): pass"
    expected = "fn takes_bar[bar: fn[x: Int]() -> Int]():\n    pass\n"
    assert_mojo_format(source, expected)
