# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.serve.pipelines.stop_detection import StopDetector


def test_stop_none():
    sd = StopDetector(None)

    assert sd.step("abc") == None


def test_stop_list():
    sd = StopDetector(["abc", "abcdef"])

    assert sd.step("a") == None
    assert sd.step("b") == None
    assert sd.step("c") == "abc"


def test_stop_str():
    sd = StopDetector("abc")

    assert sd.step("all good here") == None
    assert sd.step("ab") == None
    assert sd.step("c") == "abc"


def test_long_continuation():
    sd = StopDetector("abc")

    for x in "long continuation" * 1024:
        assert sd.step(x) == None

    assert sd.step("abc") == "abc"
