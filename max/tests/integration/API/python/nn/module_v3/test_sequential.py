# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.module_v3.Sequential."""

from __future__ import annotations

import re

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Module, Sequential, module_dataclass


@module_dataclass
class TestModule(Module):
    a: int

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.a


def strip_margin(s: str, margin_character: str = "|"):
    return re.sub(
        rf"^\s*\{margin_character}", "", s.strip(), flags=re.MULTILINE
    )


def test_repr():
    s = Sequential(
        Sequential(
            TestModule(1),
            TestModule(2),
        ),
        Sequential(
            TestModule(3),
            TestModule(4),
        ),
    )
    expected_repr = strip_margin("""
    |Sequential(
    |    Sequential(TestModule(a=1), TestModule(a=2)),
    |    Sequential(TestModule(a=3), TestModule(a=4))
    |)
    """)
    assert expected_repr == repr(s)


def test_children():
    c1 = Sequential(TestModule(1), TestModule(2))
    c2 = Sequential(TestModule(3), TestModule(4))
    s = Sequential(c1, c2)
    assert dict(s.children) == {"0": c1, "1": c2}


def test_descendents():
    t1 = TestModule(1)
    t2 = TestModule(2)
    t3 = TestModule(3)
    t4 = TestModule(4)
    c1 = Sequential(t1, t2)
    c2 = Sequential(t3, t4)
    s = Sequential(c1, c2)
    assert dict(s.descendents) == {
        "0": c1,
        "1": c2,
        "0.0": t1,
        "0.1": t2,
        "1.0": t3,
        "1.1": t4,
    }


def test_call():
    s = Sequential(
        Sequential(
            TestModule(1),
            TestModule(2),
        ),
        Sequential(
            TestModule(3),
            TestModule(4),
        ),
    )
    t = Tensor.constant(0)
    assert s(t).item() == 10
