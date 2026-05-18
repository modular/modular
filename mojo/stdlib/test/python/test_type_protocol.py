# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Integration test for TypeProtocolBuilder (tp_richcompare — all six operators)."""

from collections.abc import Callable
from typing import Any

import type_protocol_mojo_module as mojo_module  # type: ignore[import-not-found]


def _run_richcompare_assertions(new_fn: Callable[..., Any]) -> None:
    lo = new_fn(1.0)
    hi = new_fn(5.0)
    eq = new_fn(1.0)

    # __lt__ (Py_LT)
    assert lo < hi
    assert not hi < lo
    assert not lo < eq

    # __le__ (Py_LE)
    assert lo <= hi
    assert lo <= eq
    assert not hi <= lo

    # __eq__ (Py_EQ)
    assert lo == eq
    assert not lo == hi  # noqa: SIM201

    # __ne__ (Py_NE)
    assert lo != hi
    assert not lo != eq  # noqa: SIM202

    # __gt__ (Py_GT)
    assert hi > lo
    assert not lo > hi
    assert not lo > eq

    # __ge__ (Py_GE)
    assert hi >= lo
    assert lo >= eq
    assert not lo >= hi

    # Sorting relies on __lt__
    boxes = [new_fn(3.0), new_fn(1.0), new_fn(2.0)]
    boxes.sort()
    assert boxes[0].get_value() == 1.0
    assert boxes[1].get_value() == 2.0
    assert boxes[2].get_value() == 3.0

    # NotImplemented from tp_richcompare. The Box rich_compare implementation
    # returns NotImplemented when self.value == 42, so CPython tries the
    # reflected comparison on the other operand with the operator swapped
    # (e.g. `LT` becomes `GT`). Same-type two-Box case: the reflected slot
    # is called with `(other, magic)`, where `other.value == 1` so the
    # magic check doesn't fire — comparison succeeds with a normal answer.
    magic = new_fn(42.0)
    other = new_fn(1.0)
    # `magic < other`: LHS NotImplemented, reflected `other > magic` returns
    # `1.0 > 42.0` == False.
    assert not (magic < other)
    # Symmetrically `magic > other` is reflected to `other < magic` →
    # `1.0 < 42.0` == True.
    assert magic > other

    # Cross-type comparison: both sides return NotImplemented (LHS magic
    # fires OR downcast fails; RHS is `int` which doesn't know `Box`).
    # CPython falls back to identity for `==`/`!=`; ordering operators
    # raise TypeError.
    assert not (new_fn(1.0) == 5)  # noqa: SIM201 downcast failure → NotImplemented → identity
    assert new_fn(1.0) != 5
    try:
        _ = magic < 5
        raise Exception("TypeError expected for magic < 5")
    except TypeError:
        pass


def test_type_protocol() -> None:
    print("Testing type protocol (rich comparison)...")
    _run_richcompare_assertions(mojo_module.Box.new)
    print("  ptr-receiver: ok")
    _run_richcompare_assertions(mojo_module.BoxV.new)
    print("  value-receiver: ok")
    print("Type protocol tests passed!")


if __name__ == "__main__":
    test_type_protocol()
