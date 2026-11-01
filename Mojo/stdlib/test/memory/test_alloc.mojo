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

from std.memory import unsafe_destroy_n
from std.memory.alloc import (
    alloc,
    dealloc,
    Allocation,
    ManagedAllocation,
    ThinAllocation,
    Layout,
)
from std.sys import align_of, size_of

from test_utils import ObservableDel, check_niches, check_write_to
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)


def test_layout_with_default_alignment_uses_type_alignment() raises:
    var layout = Layout[Int32](length=4)
    assert_equal(len(layout), 4)
    assert_equal(type_of(layout).alignment.bytes(), align_of[Int32]())


def test_layout_with_explicit_alignment() raises:
    var layout = Layout[Int32, alignment=.of_bytes[128]()](length=4)
    assert_equal(len(layout), 4)
    assert_equal(type_of(layout).alignment.bytes(), 128)


def test_layout_single_has_count_one_and_default_alignment() raises:
    var layout = Layout[Int64].single()
    assert_equal(len(layout), 1)
    assert_equal(type_of(layout).alignment.bytes(), align_of[Int64]())


def test_layout_as_byte_layout_scales_count_and_preserves_alignment() raises:
    var layout = Layout[Int32, alignment=.of_bytes[64]()](length=4)
    var byte_layout = layout.as_byte_layout()
    assert_equal(len(byte_layout), 4 * size_of[Int32]())
    assert_equal(type_of(byte_layout).alignment.bytes(), 64)


def test_layout_write_to_and_repr() raises:
    var layout = Layout[Int, alignment=.of_bytes[64]()](length=8)
    check_write_to(
        layout,
        expected="Layout[SIMD[DType.int, 1], alignment=64](length=8)",
        is_repr=False,
    )
    check_write_to(
        layout,
        expected="Layout[SIMD[DType.int, 1], alignment=64](length=8)",
        is_repr=True,
    )


def test_alloc_and_free_round_trip_reads_and_writes_values() raises:
    var layout = Layout[Int](length=5)
    var a = alloc(layout)
    var ptr = a.unsafe_ptr()
    for i in range(5):
        ptr.unsafe_offset(i).write(i)
    # `assert_equal` can raise, and an `Allocation` must be consumed on
    # every path (including the raising one). So accumulate into a plain `Bool`,
    # `dealloc` the handle, and only then assert.
    var all_match = True
    for i in range(5):
        all_match = all_match and (ptr[unsafe_offset=i] == i)
    dealloc(a^)
    assert_true(all_match)


def test_alloc_returns_pointer_meeting_layout_alignment() raises:
    var layout = Layout[UInt8, alignment=.of_bytes[128]()](length=1)
    var a = alloc(layout)
    var addr = Int(a.unsafe_ptr())
    dealloc(a^)
    assert_equal(addr % 128, 0)


def test_alloc_with_layout_single_supports_one_element() raises:
    var layout = Layout[Int64].single()
    var a = alloc(layout)
    var ptr = a.unsafe_ptr()
    ptr.write(42)
    var value = ptr[]
    dealloc(a^)
    assert_equal(value, 42)


def test_allocation_unsafe_span_covers_layout_count() raises:
    var a = alloc(Layout[Int](length=3))
    var ptr = a.unsafe_ptr()
    for i in range(3):
        ptr.unsafe_offset(i).write(i * 10)
    var span = a.unsafe_span()
    var span_len = len(span)
    var first = span[0]
    var last = span[2]
    dealloc(a^)
    assert_equal(span_len, 3)
    assert_equal(first, 0)
    assert_equal(last, 20)


def test_allocation_count_matches_layout() raises:
    var a = alloc(Layout[Int32](length=7))
    var element_count = len(a.layout())
    var allocation_len = len(a)
    dealloc(a^)
    assert_equal(element_count, 7)
    assert_equal(allocation_len, 7)


def test_allocation_unsized_then_unsafe_with_layout_round_trip() raises:
    var layout = Layout[Int](length=4)
    var a = alloc(layout)
    a.unsafe_ptr().write(99)
    var thin = a^.into_thin()
    var value = thin.unsafe_ptr()[]
    dealloc(thin^.unsafe_with_layout(layout))
    assert_equal(value, 99)


def test_allocation_unsafe_leak_then_reconstruct() raises:
    var layout = Layout[Int](length=2)
    var ptr = alloc(layout).unsafe_leak()
    ptr.write(5)
    ptr.unsafe_offset(1).write(6)
    var total = ptr[unsafe_offset=0] + ptr[unsafe_offset=1]
    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))
    assert_equal(total, 11)


def test_thin_allocation_unsafe_with_layout_and_unsafe_ptr() raises:
    var layout = Layout[Int](length=1)
    var thin = ThinAllocation(unsafe_owned_ptr=alloc(layout).unsafe_leak())
    thin.unsafe_ptr().write(7)
    var value = thin.unsafe_ptr()[]
    dealloc(thin^.unsafe_with_layout(layout))
    assert_equal(value, 7)


def test_deletable_allocation_into_allocation_round_trip() raises:
    var layout = Layout[Int](length=3)
    var allocation = alloc(layout)
    var alloc_addr = Int(allocation.unsafe_ptr())

    # Allocation -> ManagedAllocation -> Allocation (address stays stable).
    var deletable = allocation^.into_managed()
    var deletable_addr = Int(deletable.unsafe_ptr())

    var recovered = Allocation(deletable^)
    var recovered_addr = Int(recovered.unsafe_ptr())

    # `into_allocation` hands back an non-implicitly-deletable handle: dealloc
    # it before the (raising) asserts so it can't be abandoned on a throw.
    dealloc(recovered^)

    assert_equal(deletable_addr, alloc_addr)
    assert_equal(recovered_addr, alloc_addr)


def test_deletable_allocation_layout_matches() raises:
    var deletable = ManagedAllocation(alloc(Layout[Int32](length=7)))
    assert_equal(len(deletable.layout()), 7)
    assert_equal(len(deletable), 7)
    assert_equal(
        type_of(deletable.layout()).alignment.bytes(), align_of[Int32]()
    )


def test_deletable_allocation_auto_deallocs_at_last_use() raises:
    var total = 0
    for i in range(3):
        var deletable = alloc(Layout[Int](length=1)).into_managed()
        deletable.unsafe_ptr().write(i)
        total += deletable.unsafe_ptr()[]
    assert_equal(total, 0 + 1 + 2)


def test_dealloc_does_not_run_pointee_destructors() raises:
    var deleted = False
    var obs = ObservableDel(Pointer(to=deleted).as_unsafe_any_origin())
    var a = alloc(Layout[type_of(obs)](length=1))
    a.unsafe_ptr().unsafe_write(obs^)
    dealloc(a^)
    assert_false(deleted)


def test_destroy_n_runs_pointee_destructors_before_dealloc() raises:
    var deleted = False
    var obs = ObservableDel(Pointer(to=deleted).as_unsafe_any_origin())
    var a = alloc(Layout[type_of(obs)](length=1))
    a.unsafe_ptr().unsafe_write(obs^)
    unsafe_destroy_n(a.unsafe_ptr(), 1)
    var ran = deleted
    dealloc(a^)
    assert_true(ran)


def test_alloc_count_0() raises:
    comptime ZST = Array[Int, 0]
    var zst_alloc = alloc(Layout[ZST](length=0)).into_managed()
    assert_equal(len(zst_alloc), 0)
    dealloc(zst_alloc^)

    var int_alloc = alloc(Layout[Int](length=0)).into_managed()
    assert_equal(len(int_alloc), 0)
    dealloc(int_alloc^)


def test_alloc_free_single_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](length=1)
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(
        ptr.unsafe_bitcast[Int]().as_unsafe_any_origin(),
        ptr[].unsafe_ptr().as_unsafe_any_origin(),
    )

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_single_zst_lifecycle() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](length=1)
    var alloc = alloc(layout)
    var ptr = alloc^.unsafe_leak()

    ptr.unsafe_write(ZST(fill=0))
    assert_equal(0, len(ptr[]))
    ptr.unsafe_deinit_pointee()
    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_alloc_free_many_zst() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](
        length=Int.MAX
    )  # It's a ZST, it doesn't take memory
    var ptr = alloc(layout).unsafe_leak()

    assert_equal(ptr.unsafe_bitcast[Int](), ptr[].unsafe_ptr())

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_many_zst_lifecycle() raises:
    comptime ZST = Array[Int, 0]
    comptime assert (
        size_of[ZST]() == 0
    ), "Please find a ZST to use for this test."

    var layout = Layout[ZST](length=Int.MAX)
    var ptr = alloc(layout).unsafe_leak()

    ptr.unsafe_bitcast[Array[ZST, Int.MAX]]().unsafe_write({fill = ZST(fill=0)})

    assert_equal(0, len(ptr[]))
    assert_equal(0, len(ptr[unsafe_offset=Int.MAX]))

    ptr.unsafe_bitcast[Array[ZST, Int.MAX]]().unsafe_deinit_pointee()

    dealloc(ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout))


def test_layout_niche() raises:
    comptime LayoutType = Layout[Int64]
    assert_equal(LayoutType.niche_count(), Int.MAX)
    assert_equal(size_of[Optional[LayoutType]](), size_of[LayoutType]())
    check_niches[LayoutType]()

    var some = Optional(LayoutType(length=4))
    assert_true(some)
    assert_equal(len(some.value()), 4)

    var empty = Optional(LayoutType(length=0))
    assert_true(empty)
    assert_equal(len(empty.value()), 0)

    var none = Optional[LayoutType](None)
    assert_false(none)


def test_allocation_niche() raises:
    comptime AllocationType = Allocation[Int64]
    assert_equal(AllocationType.niche_count(), Int.MAX)
    assert_equal(size_of[Optional[AllocationType]](), size_of[AllocationType]())
    check_niches[AllocationType]()

    def _dealloc(var allocation: AllocationType):
        dealloc(allocation^)

    # `Optional[Allocation]` can't be abandoned by a raising assert, so read
    # each one and destroy it before asserting.
    var some = Optional(alloc(Layout[Int64](length=4)))
    var some_engaged = Bool(some)
    var some_length = len(some.value())
    some^.deinit_with(_dealloc)
    assert_true(some_engaged)
    assert_equal(some_length, 4)

    var empty = Optional(alloc(Layout[Int64](length=0)))
    var empty_engaged = Bool(empty)
    empty^.deinit_with(_dealloc)
    assert_true(empty_engaged)

    var none = Optional[AllocationType](None)
    var none_engaged = Bool(none)
    none^.deinit_assert_empty()
    assert_false(none_engaged)


def test_managed_allocation_niche() raises:
    comptime ManagedType = ManagedAllocation[Int64]
    assert_equal(ManagedType.niche_count(), Int.MAX)
    assert_equal(size_of[Optional[ManagedType]](), size_of[ManagedType]())
    check_niches[ManagedType]()

    var some = Optional(alloc(Layout[Int64](length=4)).into_managed())
    assert_true(some)
    assert_equal(len(some.value()), 4)

    var none = Optional[ManagedType](None)
    assert_false(none)


def test_thin_allocation_niche() raises:
    comptime ThinType = ThinAllocation[Int64]
    assert_equal(ThinType.niche_count(), 1)
    assert_equal(size_of[Optional[ThinType]](), size_of[ThinType]())
    check_niches[ThinType]()

    var layout = Layout[Int64](length=4)

    def _dealloc(var thin: ThinType) {layout}:
        dealloc(thin^.unsafe_with_layout(layout))

    var some = Optional(alloc(layout).into_thin())
    var some_engaged = Bool(some)
    some^.deinit_with(_dealloc)
    assert_true(some_engaged)

    var none = Optional[ThinType](None)
    var none_engaged = Bool(none)
    none^.deinit_assert_empty()
    assert_false(none_engaged)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
