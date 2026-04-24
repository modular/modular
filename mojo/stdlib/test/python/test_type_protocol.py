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

import type_protocol_mojo_module as mojo_module  # type: ignore[import-not-found]


def _run_richcompare_assertions(new_fn) -> None:
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
    assert not lo == hi

    # __ne__ (Py_NE)
    assert lo != hi
    assert not lo != eq

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


def test_type_protocol() -> None:
    print("Testing type protocol (rich comparison)...")
    _run_richcompare_assertions(mojo_module.Box.new)
    print("  ptr-receiver: ok")
    _run_richcompare_assertions(mojo_module.BoxV.new)
    print("  value-receiver: ok")
    print("Type protocol tests passed!")


if __name__ == "__main__":
    test_type_protocol()
