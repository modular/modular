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

"""Integration test for NumberProtocolBuilder (unary, bool, conversion, binary, ternary slots)."""

import operator

import number_mojo_module as mojo_module  # type: ignore[import-not-found]


def _run_number_assertions(new_fn, val_fn) -> None:
    n = new_fn

    # __neg__ (nb_negative)
    assert val_fn(-n(7)) == -7
    assert val_fn(-n(-3)) == 3

    # __abs__ (nb_absolute)
    assert val_fn(abs(n(-5))) == 5
    assert val_fn(abs(n(5))) == 5

    # __pos__ (nb_positive)
    assert val_fn(+n(4)) == 4

    # __invert__ (nb_invert)
    assert val_fn(~n(0)) == -1
    assert val_fn(~n(6)) == -7

    # __bool__ (nb_bool)
    assert bool(n(1))
    assert bool(n(-1))
    assert not bool(n(0))

    # __int__ (nb_int)
    assert int(n(42)) == 42

    # __float__ (nb_float)
    assert float(n(3)) == 3.0

    # __index__ (nb_index)
    assert operator.index(n(10)) == 10
    lst = [0, 1, 2, 3]
    assert lst[n(2)] == 2

    # __add__ (nb_add)
    assert val_fn(n(3) + n(4)) == 7

    # __add__ with non-type returns NotImplemented → TypeError
    try:
        _ = n(1) + 42
        raise Exception("TypeError expected")
    except TypeError:
        pass

    # __sub__ (nb_subtract)
    assert val_fn(n(10) - n(3)) == 7

    # __mul__ (nb_multiply)
    assert val_fn(n(6) * n(7)) == 42

    # __floordiv__ (nb_floor_divide)
    assert val_fn(n(17) // n(5)) == 3

    # __mod__ (nb_remainder)
    assert val_fn(n(17) % n(5)) == 2

    # __and__ (nb_and)
    assert val_fn(n(0b1100) & n(0b1010)) == 0b1000

    # __or__ (nb_or)
    assert val_fn(n(0b1100) | n(0b1010)) == 0b1110

    # __xor__ (nb_xor)
    assert val_fn(n(0b1100) ^ n(0b1010)) == 0b0110

    # __lshift__ (nb_lshift)
    assert val_fn(n(1) << n(4)) == 16

    # __rshift__ (nb_rshift)
    assert val_fn(n(32) >> n(2)) == 8

    # __pow__ (nb_power)
    assert val_fn(n(2) ** n(10)) == 1024
    assert val_fn(n(3) ** n(3)) == 27


def _run_inplace_assertions(new_fn, val_fn) -> None:
    n = new_fn(10)
    m = new_fn(3)

    n += m
    assert val_fn(n) == 13

    n -= m
    assert val_fn(n) == 10

    n *= m
    assert val_fn(n) == 30

    n **= new_fn(2)
    assert val_fn(n) == 900


def test_number_protocol() -> None:
    print("Testing number protocol...")
    _run_number_assertions(mojo_module.Number.new, lambda x: x.get_value())
    print("  ptr-receiver: ok")
    _run_number_assertions(mojo_module.NumberV.new, lambda x: x.get_value())
    print("  value-receiver: ok")
    _run_inplace_assertions(mojo_module.NumberM.new, lambda x: x.get_value())
    print("  mut-receiver in-place: ok")
    print("Number protocol tests passed!")


if __name__ == "__main__":
    test_number_protocol()
