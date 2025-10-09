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

from math import ceildiv

from buffer.dimlist import Dim, DimList
from internal_utils._utils import ValOrDim, dynamic, static
from testing import TestSuite, assert_equal, assert_false, assert_true


# CHECK-LABEL: test_dim_list
def test_dim_list():
    print("== test_dim_list")

    var lst0 = DimList(1, 2, 3, 4)
    var lst1 = DimList(Dim(), 2, 3, 4)

    # CHECK: [1, 2, 3, 4]
    print(lst0)

    # CHECK: 24
    print(lst0.product[4]().get())

    assert_equal(lst0.product[3, 4](), 4)

    # CHECK: True
    print(lst0.all_known[4]())

    # CHECK: False
    print(lst1.all_known[4]())

    # CHECK: True
    print(lst1.all_known[1, 4]())

    # CHECK: False
    print(lst1.has_value[0]())

    # CHECK: True
    print(lst1.has_value[2]())

    assert_equal(lst0.product(), 1 * 2 * 3 * 4)

    assert_equal(lst1.product(), Dim())


# CHECK-LABEL: test_dim
def test_dim():
    print("== test_dim")

    var dim0 = Dim(8)
    # CHECK: True
    print(dim0.is_multiple[4]())

    var dim1 = Dim()
    # CHECK: False
    print(dim1.is_multiple[4]())

    var dim2 = dim0 // 2
    # CHECK: True
    print(dim2.has_value())
    # CHECK: 4
    print(dim2.get())

    var dim3 = dim1 // Dim()
    # CHECK: False
    print(dim3.has_value())


def test_dim_to_string():
    assert_equal(String(Dim()), "?")
    assert_equal(String(Dim(33)), "33")
    assert_equal(String(DimList(2, Dim(), 3)), "[2, ?, 3]")
    assert_equal(String(DimList.create_unknown[5]()), "[?, ?, ?, ?, ?]")


def test_dimlist_repr():
    assert_equal(repr(DimList(2, Dim(), 3)), "DimList([2, ?, 3])")
    assert_equal(repr(DimList.create_unknown[5]()), "DimList([?, ?, ?, ?, ?])")


def test_dimlist_eq():
    assert_true(DimList(Dim(), 42, Dim()) == DimList(Dim(), 42, Dim()))
    assert_true(DimList(Dim(), Dim()) == DimList(Dim(), Dim()))
    assert_true(DimList() == DimList())
    assert_true(DimList(1, 2, 3) == DimList(1, 2, 3))

    assert_false(DimList(Dim(), 42, 41) == DimList(Dim(), 42, Dim()))
    assert_false(DimList(Dim()) == DimList())
    assert_false(DimList(1, 2, Dim()) == DimList(1, 2, 3))
    assert_false(
        DimList(1, 2, Dim())
        == DimList(
            1,
            2,
        )
    )
    assert_false(
        DimList(
            1,
            2,
        )
        == DimList(1, 2, Dim())
    )


fn test_dim_ceildiv() raises:
    fn test_dim_ceildiv(m: ValOrDim) -> Dim:
        alias BLOCK_SCALE_M = 128
        return ceildiv(m.dim, BLOCK_SCALE_M)

    assert_equal(String(test_dim_ceildiv(dynamic(120))), "?")
    assert_equal(String(test_dim_ceildiv(static[120]())), "1")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
