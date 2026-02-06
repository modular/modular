# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest

import mblack
from mblack.parsing import InvalidInput
from tests.util import assert_mojo_format


def test_unknown_convention_raises_error():
    source = "fn foo(aaa self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


def test_inout_raises_error():
    """inout has been removed from Mojo; the formatter should reject it."""
    source = "fn method(inout self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


def test_borrowed_raises_error():
    """borrowed has been removed from Mojo; the formatter should reject it."""
    source = "fn method(borrowed self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


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
