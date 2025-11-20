# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tests.util import assert_mojo_format


def test_var_keyword_spacing_simple():
    source = "fn method(var self): pass"
    expected = (
        "fn method(var self):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_with_type():
    source = "fn func(var x: Int): pass"
    expected = (
        "fn func(var x: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_mixed():
    source = "fn method(var self, y: Int): pass"
    expected = (
        "fn method(var self, y: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing():
    source = "fn __init__(out self, mut v:Int, read x:Int): pass"
    expected = (
        "fn __init__(out self, mut v: Int, read x: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing_variadics():
    source =  (
        "fn __init__(out self, mut *v: Int, read *x: *Int, mut **kwargs: Int): "
        "pass"
    )
    expected = (
        "fn __init__(out self, mut *v: Int, read *x: *Int, mut **kwargs: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
