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
"""Checks native runtime-layout memory mapping against legacy scalar views.

Masked and gathered tensors have additional access contracts, and structured
legacy elements have different storage units. These tests establish equivalence
only for unmasked, ungathered scalar storage with explicit matching strides.
Nested layouts are constructed directly, not through the lossy product_each
conversion in lt_to_tt. Overflow checks never dereference their offsets.
"""

from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tile_layout import Layout as NativeLayout
from std.testing import TestSuite, assert_equal


def _check_flat[
    Native: TensorLayout,
    //,
    legacy_layout: Layout,
    index_type: DType,
](native_layout: Native, row_stride: Int, col_stride: Int) raises:
    var legacy_storage = Array[Int32, 49](fill=-777)
    var native_storage = Array[Int32, 49](fill=-777)
    var expected = Array[Int32, 49](fill=-777)
    var runtime = RuntimeLayout[
        legacy_layout, element_type=.int32, linear_idx_type=index_type
    ](
        RuntimeTuple[legacy_layout.shape, element_type=.int32](3, 4),
        RuntimeTuple[legacy_layout.stride, element_type=index_type](
            row_stride, col_stride
        ),
    )
    # Neither tensor owns or extends the owner's lifetime. The interior
    # base leaves redzones on both sides of the strided view.
    var legacy = LayoutTensor[
        .int32, legacy_layout, linear_idx_type=index_type
    ](legacy_storage.unsafe_ptr() + 16, runtime)
    var native = TileTensor[linear_idx_type=index_type](
        native_storage.unsafe_ptr() + 16, native_layout
    )
    var tracked: MutPointer[
        Int32, origin_of(native_storage)
    ] = native.unsafe_ptr()
    assert_equal(Int(tracked), Int(native_storage.unsafe_ptr() + 16))
    comptime assert native.flat_rank == 2
    comptime assert native.address_space == .GENERIC
    comptime assert native.element_size == 1
    assert_equal(Int(native.dim[0]()), 3)
    assert_equal(Int(native.dim[1]()), 4)
    for i in range(3):
        for j in range(4):
            var crd = Coord(Scalar[index_type](i), Scalar[index_type](j))
            var offset = i * row_stride + j * col_stride
            assert_equal(
                Int(native.layout[linear_idx_type=index_type](crd)), offset
            )
            assert_equal(
                Int(runtime(RuntimeTuple[IntTuple(-1, -1)](i, j))), offset
            )
            var value = Int32(100 * i + j + 1)
            legacy[i, j] = value
            native.store(crd, value)
            expected[16 + offset] = value
            assert_equal(native.load(crd), legacy.load_scalar(i, j))
            assert_equal(native.ptr_at_offset(crd).load(), value)
    # Comparing every slot detects writes into padding and both redzones.
    for i in range(49):
        assert_equal(legacy_storage[i], expected[i])
        assert_equal(native_storage[i], expected[i])


def test_static_padded_and_column_major() raises:
    _check_flat[Layout(IntTuple(3, 4), IntTuple(6, 1)), index_type=.int32](
        NativeLayout(Coord(Idx[3], Idx[4]), Coord(Idx[6], Idx[1])), 6, 1
    )
    _check_flat[Layout(IntTuple(3, 4), IntTuple(1, 4)), index_type=.int64](
        NativeLayout(Coord(Idx[3], Idx[4]), Coord(Idx[1], Idx[4])), 1, 4
    )


def test_dynamic_and_mixed_strides() raises:
    comptime dynamic = Layout(
        IntTuple(UNKNOWN_VALUE, 4), IntTuple(UNKNOWN_VALUE, 1)
    )
    for stride in range(4, 7):
        _check_flat[dynamic, index_type=.int32](
            NativeLayout(Coord(Int32(3), Idx[4]), Coord(Int32(stride), Idx[1])),
            stride,
            1,
        )
    _check_flat[dynamic, index_type=.int64](
        NativeLayout(Coord(Int64(3), Idx[4]), Coord(Int64(6), Idx[1])), 6, 1
    )


def test_nested_runtime_colex_memory_mapping() raises:
    comptime nested = IntTuple(IntTuple(-1, -1), IntTuple(-1, -1))
    comptime legacy_layout = Layout(nested, nested)
    var runtime = RuntimeLayout[legacy_layout](
        RuntimeTuple[legacy_layout.shape](2, 3, 2, 2),
        RuntimeTuple[legacy_layout.stride](1, 8, 24, 4),
    )
    var storage = Array[Int32, 49](fill=-777)
    var expected = Array[Int32, 49](fill=-777)
    var legacy = LayoutTensor[.int32, legacy_layout](
        storage.unsafe_ptr() + 1, runtime
    )
    var native = TileTensor(
        storage.unsafe_ptr() + 1,
        NativeLayout(
            Coord(Coord(Int32(2), Idx[3]), Coord(Idx[2], Int32(2))),
            Coord(Coord(Idx[1], Int32(8)), Coord(Int32(24), Idx[4])),
        ),
    )
    comptime assert native.rank == 2 and native.flat_rank == 4
    # A scalar per outer mode expands first-submode-fastest, independently
    # of the physical stride order. It is not a row-major reshape.
    for m in range(6):
        for n in range(4):
            var offset = m % 2 + (m // 2) * 8 + (n % 2) * 24 + (n // 2) * 4
            var crd = Coord(Int32(m), Int32(n))
            var hierarchical = Coord(
                Coord(Int32(m % 2), Int32(m // 2)),
                Coord(Int32(n % 2), Int32(n // 2)),
            )
            var old_crd = RuntimeTuple[IntTuple(-1, -1)](m, n)
            assert_equal(Int(runtime(old_crd)), offset)
            assert_equal(Int(native.layout(crd)), offset)
            assert_equal(Int(native.layout(hierarchical)), offset)
            var value = Int32(100 * m + n + 1)
            native.store(hierarchical, value)
            expected[1 + offset] = value
            assert_equal(legacy.load_scalar(old_crd), value)
            assert_equal(native.load(crd), value)
    for i in range(49):
        assert_equal(storage[i], expected[i])


def test_typed_index_wrap_is_arithmetic_only() raises:
    comptime legacy_layout = Layout(IntTuple(3, 4), IntTuple(-1, 1))
    var native = NativeLayout(
        Coord(Idx[3], Idx[4]), Coord(Int64(2147483647), Idx[1])
    )
    var runtime = RuntimeLayout[
        legacy_layout, element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[legacy_layout.shape, element_type=.int32](3, 4),
        RuntimeTuple[legacy_layout.stride, element_type=.int32](2147483647, 1),
    )
    var crd = Coord(Int32(2), Int32(3))
    var wrapped = native[linear_idx_type=.int32](crd)
    var widened = native[linear_idx_type=.int64](crd)
    assert_equal(Int(wrapped), 1)
    assert_equal(Int(widened), 4294967297)
    assert_equal(
        Int(runtime(RuntimeTuple[IntTuple(-1, -1)](2, 3))), Int(wrapped)
    )


def test_nested_index_narrowing_precedes_colex_expansion() raises:
    comptime old = Layout(
        IntTuple(IntTuple(3, 2), 4), IntTuple(IntTuple(1, 8), 32)
    )
    var runtime = RuntimeLayout[old, linear_idx_type=.int32]()
    var native = NativeLayout(
        Coord(Coord(Idx[3], Idx[2]), Idx[4]),
        Coord(Coord(Idx[1], Idx[8]), Idx[32]),
    )
    var wide = Int64(4294967297)
    var narrowed = RuntimeTuple[IntTuple(-1, -1), element_type=.int32](
        Int(Int32(wide)), 0
    )
    # crd2idx splits a scalar coordinate at the output dtype, so a narrow
    # output dtype narrows before the nested divmod, like legacy __getitem__.
    assert_equal(Int(runtime(narrowed)), 1)
    assert_equal(
        Int(native[linear_idx_type=.int32](Coord(Int32(wide), Int32(0)))), 1
    )
    assert_equal(Int(native[linear_idx_type=.int32](Coord(wide, Int64(0)))), 1)


def test_borrowed_views_keep_origin_and_alias_storage() raises:
    var storage = Array[Int32, 14](fill=-777)
    var tensor = TileTensor(storage.unsafe_ptr() + 1, row_major[3, 4]())
    var reshaped = tensor.reshape[2, 6]()
    var tracked: MutPointer[Int32, origin_of(storage)] = reshaped.unsafe_ptr()
    assert_equal(Int(tracked), Int(storage.unsafe_ptr() + 1))
    comptime assert reshaped.Engine == tensor.Engine
    comptime assert reshaped.address_space == tensor.address_space
    for i in range(12):
        tensor.store(Coord(i // 4, i % 4), Int32(i + 1))
        assert_equal(reshaped.load(Coord(i // 6, i % 6)), Int32(i + 1))
    assert_equal(storage[0], -777)
    assert_equal(storage[13], -777)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
