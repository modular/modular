# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# ===----------------------------------------------------------------------=== #

from testing import assert_equal, TestSuite
from format._utils import Repr


def test_span_repr_nonempty():
    var l = [1, 2, 3]
    var s = Span(l)[:2]
    # Repr should use the writable repr implementation (non-reflective)
    assert_equal(String(Repr(s)), "Span[mut=True, Int](2)")


def test_span_repr_empty():
    var a = InlineArray[Int, 1](0)
    var s = Span(a)[:0]
    assert_equal(String(Repr(s)), "Span[mut=True, Int](0)")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
