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

from layout import (
    Coord,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout._utils import _get_bounds
from layout.tile_layout import Layout as TileLayout
from std.testing import assert_equal


def legacy_oracle(tensor: LayoutTensor) -> Int:
    if tensor.dim[0]() == 0 or tensor.dim[1]() == 0:
        return 0
    var strides = tensor.runtime_layout.stride.value
    return (
        tensor._get_offset(
            strides,
            type_of(tensor).idx_list_t[2](
                tensor.dim[0]() - 1, tensor.dim[1]() - 1
            ),
        )
        + 1
    )


def native_oracle(tensor: TileTensor) -> Int:
    var m = Int(tensor.dim[0]())
    var n = Int(tensor.dim[1]())
    if m <= 0 or n <= 0:
        return 0
    return (
        (m - 1) * Int(tensor.layout.stride[0]().value())
        + (n - 1) * Int(tensor.layout.stride[1]().value())
        + 1
    )


def check_dynamic[
    index_type: DType, space: AddressSpace
](m: Int, n: Int, stride0: Int, stride1: Int) raises:
    comptime legacy_layout = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
    )
    var runtime = RuntimeLayout[
        legacy_layout, element_type=.int64, linear_idx_type=index_type
    ](
        RuntimeTuple[legacy_layout.shape, element_type=.int64](m, n),
        RuntimeTuple[legacy_layout.stride, element_type=index_type](
            stride0, stride1
        ),
    )
    # These views are never dereferenced; only their layout metadata is read.
    var ptr = Pointer[
        Float32, MutAnyOrigin, address_space=space
    ].unsafe_dangling()
    var legacy = LayoutTensor[
        .float32,
        legacy_layout,
        MutAnyOrigin,
        address_space=space,
        layout_int_type=.int64,
        linear_idx_type=index_type,
    ](ptr, runtime)
    assert_equal(_get_bounds(legacy), legacy_oracle(legacy))
    var native = TileTensor[address_space=space, linear_idx_type=.int64](
        ptr, TileLayout(Coord(m, n), Coord(stride0, stride1))
    )
    assert_equal(_get_bounds(native), native_oracle(native))


def check_static_masked[space: AddressSpace]() raises:
    comptime shape = Layout.row_major(4, 8)
    var ptr = Pointer[
        Float32, MutAnyOrigin, address_space=space
    ].unsafe_dangling()
    var static_tensor = LayoutTensor[
        .float32, shape, MutAnyOrigin, address_space=space
    ](ptr)
    assert_equal(_get_bounds(static_tensor), 32)
    var runtime = RuntimeLayout[shape](
        RuntimeTuple[shape.shape](2, 3), RuntimeTuple[shape.stride](8, 1)
    )
    var masked = LayoutTensor[
        .float32, shape, MutAnyOrigin, address_space=space, masked=True
    ](ptr, runtime)
    assert_equal(_get_bounds(masked), legacy_oracle(masked))
    assert_equal(_get_bounds(masked), 11)
    runtime.shape.value[0] = 0
    var empty = type_of(masked)(ptr, runtime)
    assert_equal(_get_bounds(empty), 0)


def check_cases[index_type: DType, space: AddressSpace]() raises:
    check_dynamic[index_type, space](3, 5, 9, 2)
    check_dynamic[index_type, space](0, 5, 9, 2)
    check_dynamic[index_type, space](3, 0, 9, 2)
    check_dynamic[index_type, space](-2, 5, 9, 2)
    check_dynamic[index_type, space](3, -2, 9, 2)
    check_dynamic[index_type, space](3, 5, -9, -2)
    check_dynamic[index_type, space](65536, 1, 65536, 1)
    check_dynamic[index_type, space](65536, 65536, 65536, 65536)
    check_dynamic[index_type, space]((1 << 31) + 1, 1, 1, 1)


def check_space[space: AddressSpace]() raises:
    check_static_masked[space]()
    check_cases[.int8, space]()
    check_cases[.int16, space]()
    check_cases[.int32, space]()
    check_cases[.int64, space]()


def main() raises:
    check_space[AddressSpace.GENERIC]()
    check_space[AddressSpace.SHARED]()
