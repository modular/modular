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
    source = "def foo(aaa self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


def test_inout_raises_error():
    """inout has been removed from Mojo; the formatter should reject it."""
    source = "def method(inout self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


def test_borrowed_raises_error():
    """borrowed has been removed from Mojo; the formatter should reject it."""
    source = "def method(borrowed self): pass"
    mode = mblack.Mode(target_versions={mblack.TargetVersion.MOJO})
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mblack.format_str(source, mode=mode)


def test_var_keyword_spacing_simple():
    source = "def method(var self): pass"
    expected = (
        "def method(var self):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_with_type():
    source = "def func(var x: Int): pass"
    expected = (
        "def func(var x: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_mixed():
    source = "def method(var self, y: Int): pass"
    expected = (
        "def method(var self, y: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing():
    source = "def __init__(out self, mut v:Int, read x:Int): pass"
    expected = (
        "def __init__(out self, mut v: Int, read x: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing_variadics():
    source =  (
        "def __init__(out self, mut *v: Int, read *x: *Int, mut **kwargs: Int): "
        "pass"
    )
    expected = (
        "def __init__(out self, mut *v: Int, read *x: *Int, mut **kwargs: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
