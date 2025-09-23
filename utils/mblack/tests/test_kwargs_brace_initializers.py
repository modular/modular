# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_brace_initializer_simple():
    source = "obj{field=value}"
    expected = "obj {field = value}\n"
    assert_mojo_format(source, expected)


def test_brace_initializer_multiple_kwargs():
    source = "obj{field1=value1, field2=value2}"
    expected = "obj {field1 = value1, field2 = value2}\n"
    assert_mojo_format(source, expected)


def test_brace_initializer_multiple_kwargs_with_result():
    source = "x = SomeType{x=1, y=2}"
    expected = "x = SomeType {x = 1, y = 2}\n"
    assert_mojo_format(source, expected)


def test_brace_initializer_mixed_kwargs_with_result():
    source = "result = foo{3, y=4}"
    expected = "result = foo {3, y = 4}\n"
    assert_mojo_format(source, expected)
