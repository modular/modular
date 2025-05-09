# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from layout.int_tuple import IntTuple
from layout.int_tuple import crd2idx as crd2idx_int_tuple
from layout.int_tuple import fill_like
from layout.int_tuple import idx2crd as idx2crd_int_tuple
from layout.int_tuple import shape_div as shape_div_int_tuple
from layout.runtime_tuple import (
    UNKNOWN_VALUE,
    RuntimeTuple,
    crd2idx,
    idx2crd,
    is_int,
    is_tuple,
    prefix_product,
    shape_div,
)
from testing import assert_equal, assert_false, assert_true


def test_construct():
    print("== test_construct")
    var t1 = RuntimeTuple[IntTuple(1, 44, IntTuple(1, 102))]()
    assert_equal(String(t1.__str__()), "(1, 44, (1, 102))")

    var t2 = RuntimeTuple[IntTuple(33, IntTuple(44, IntTuple(55, 202)))]()
    assert_equal(String(t2), "(33, (44, (55, 202)))")

    var t3 = RuntimeTuple[IntTuple(UNKNOWN_VALUE, 1)]()
    assert_equal(String(t3), "(-1, 1)")


def test_concat():
    print("== test_concat")
    var lhs = RuntimeTuple[
        IntTuple(1, UNKNOWN_VALUE, IntTuple(1, UNKNOWN_VALUE))
    ](1, 44, 1, 102)
    var rhs = RuntimeTuple[
        IntTuple(
            UNKNOWN_VALUE, IntTuple(UNKNOWN_VALUE, IntTuple(UNKNOWN_VALUE, 202))
        )
    ](33, 44, 55, 202)
    print(lhs.concat(rhs))


def test_flatten():
    print("== test_flatten")
    var t1 = RuntimeTuple[IntTuple(1, 44, IntTuple(1, 102))]()
    assert_equal(String(t1.flatten()), "(1, 44, 1, 102)")


def test_prefix_product():
    print("== test_prefix_product")
    var t1 = RuntimeTuple[IntTuple(UNKNOWN_VALUE, IntTuple(2, 4))](8, 2, 4)
    var t1_p = prefix_product(t1)
    assert_equal(String(t1_p), "(1, (8, 16))")
    assert_equal(String(t1_p.S), "(1, (-1, -1))")


def test_idx2crd():
    print("== test_idx2crd")

    alias tuple = IntTuple(2, IntTuple(2, 4))

    var r_tuple = RuntimeTuple[fill_like(tuple, UNKNOWN_VALUE)](2, 2, 4)

    for i in range(16):
        assert_equal(
            String(idx2crd_int_tuple(i, tuple)),
            String(idx2crd(RuntimeTuple[UNKNOWN_VALUE](i), r_tuple)),
        )


def test_crd2idx():
    print("== test_crd2idx")
    alias shape_t = IntTuple(4, 4)
    alias stride_t = IntTuple(4, 1)
    alias unk_r2_t = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)

    for i in range(4):
        for j in range(4):
            assert_equal(
                String(
                    crd2idx(
                        RuntimeTuple[unk_r2_t](i, j),
                        RuntimeTuple[unk_r2_t](4, 4),
                        RuntimeTuple[unk_r2_t](4, 1),
                    )
                ),
                String(crd2idx_int_tuple(IntTuple(i, j), shape_t, stride_t)),
            )


def test_shape_div():
    print("== test_shape_div")
    alias shape_a_1 = IntTuple(4, 4)
    alias shape_b_1 = IntTuple(2, 1)
    var shape_a_r_1 = RuntimeTuple[fill_like(shape_a_1, UNKNOWN_VALUE)](4, 4)
    var shape_b_r_1 = RuntimeTuple[fill_like(shape_b_1, UNKNOWN_VALUE)](2, 1)
    assert_equal(
        String(shape_div(shape_a_r_1, shape_b_r_1)),
        String(shape_div_int_tuple(shape_a_1, shape_b_1)),
    )
    assert_equal(String(shape_div(shape_a_r_1, shape_b_r_1).S), "(-1, -1)")

    alias shape_a_2 = IntTuple(3, 4)
    alias shape_b_2 = 6
    var shape_a_r_2 = RuntimeTuple[fill_like(shape_a_2, UNKNOWN_VALUE)](3, 4)
    var shape_b_r_2 = RuntimeTuple[fill_like(shape_b_2, UNKNOWN_VALUE)](6)
    assert_equal(
        String(shape_div(shape_a_r_2, shape_b_r_2)),
        String(shape_div_int_tuple(shape_a_2, shape_b_2)),
    )
    assert_equal(String(shape_div(shape_a_r_2, shape_b_r_2).S), "(-1, -1)")


def main():
    test_construct()
    test_concat()
    test_flatten()
    test_prefix_product()
    test_idx2crd()
    test_crd2idx()
    test_shape_div()
