# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for the // (positional-only separator) sigil formatting in Mojo."""

from tests.util import assert_mojo_format


def test_doubleslash_on_own_line_simple():
    """Test that // is placed on its own line in multi-line function signatures."""
    source = "fn foo[a: Int, //, b: Int, *, c: Int,](): pass"
    expected = """\
fn foo[
    a: Int,
    //,
    b: Int,
    *,
    c: Int,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_on_own_line_with_types():
    """Test // with typed parameters is placed on its own line."""
    source = "fn bar[x: Int, y: String, //, z: Bool,](): pass"
    expected = """\
fn bar[
    x: Int,
    y: String,
    //,
    z: Bool,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_with_defaults():
    """Test // with default values is placed on its own line."""
    source = "fn baz[a: Int = 1, b: Int = 2, //, c: Int = 3,](): pass"
    expected = """\
fn baz[
    a: Int = 1,
    b: Int = 2,
    //,
    c: Int = 3,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_inline_fits():
    """Test that // stays inline when the signature fits on one line."""
    source = "fn f[a: Int, //](): pass"
    expected = """\
fn f[a: Int, //]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_and_star_together():
    """Test that both // and * are placed on their own lines."""
    source = "fn combined[pos1: Int, pos2: Int, //, normal: Int, *, kwonly: Int,](): pass"
    expected = """\
fn combined[
    pos1: Int,
    pos2: Int,
    //,
    normal: Int,
    *,
    kwonly: Int,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_in_long_signature():
    """Test // in a long signature gets its own line."""
    source = "fn long_function_name[very_long_parameter_name: Int, another_long_param: String, //, yet_another_param: Bool,](): pass"
    expected = """\
fn long_function_name[
    very_long_parameter_name: Int,
    another_long_param: String,
    //,
    yet_another_param: Bool,
]():
    pass
"""
    assert_mojo_format(source, expected)
