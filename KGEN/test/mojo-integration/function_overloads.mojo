# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s

from testing import assert_equal


struct OverloadedKwArgs:
    var val: Int

    fn __init__(out self, single: Int):
        self.val = single

    fn __init__(out self, *, double: Int):
        self.val = double * 2

    fn __init__(out self, *, triple: Int):
        self.val = triple * 3


def test_keyword_name_overload():
    assert_equal(OverloadedKwArgs(1).val, 1)
    assert_equal(OverloadedKwArgs(double=1).val, 2)
    assert_equal(OverloadedKwArgs(triple=1).val, 3)


struct OverloadedIndexers:
    var vals: List[Int]

    fn __init__(out self):
        self.vals = [0, 1, 2]

    fn __getitem__(self, idx: Int) -> Int:
        return self.vals[idx]

    fn __getitem__(self, *, idx2: Int) -> Int:
        return self.vals[idx2 * 2]

    fn __setitem__(mut self, idx: Int, val: Int):
        self.vals[idx] = val

    fn __setitem__(mut self, val: Int, *, idx2: Int):
        self.vals[idx2 * 2] = val


def test_indexer_overload():
    var x = OverloadedIndexers()
    assert_equal(x[1], 1)
    assert_equal(x[idx2=1], 2)

    x[1] = 42
    x[idx2=1] = 84

    assert_equal(x[1], 42)
    assert_equal(x[idx2=1], 84)


struct OverloadArgumentConventions:
    var val: Int

    fn __init__(out self, *, x: Int):
        self.val = x

    fn __init__(out self, *, mut x2: Int):
        x2 *= 2
        self.val = x2

    fn __init__(out self, *, owned x3: Int):
        self.val = x3 * 3


fn out_mut_kw_argument(*, out x: Int):
    x = 42


fn out_mut_kw_argument(*, mut x: Int):
    x = 84


def test_overloading_argument_conventions():
    var res = OverloadArgumentConventions(x=1)
    assert_equal(res.val, 1)

    var val = 1
    res = OverloadArgumentConventions(x2=val)
    assert_equal(val, 2)
    assert_equal(res.val, 2)

    res = OverloadArgumentConventions(x3=val)
    assert_equal(res.val, 6)

    var x = out_mut_kw_argument()
    assert_equal(x, 42)
    out_mut_kw_argument(x=x)
    assert_equal(x, 84)


struct KeywordOverloadDefaultArgs:
    var val: Int

    fn __init__(out self, *, x: Int = 1):
        self.val = x

    fn __init__(out self, *, x2: Int):
        self.val = x2 * 2


def test_keyword_overload_default_args():
    var res = KeywordOverloadDefaultArgs()
    assert_equal(res.val, 1)

    res = KeywordOverloadDefaultArgs(x=2)
    assert_equal(res.val, 2)

    res = KeywordOverloadDefaultArgs(x2=2)
    assert_equal(res.val, 4)


def main():
    test_keyword_name_overload()
    test_indexer_overload()
    test_overloading_argument_conventions()
    test_keyword_overload_default_args()
