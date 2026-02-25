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

from test_utils import (
    ExplicitCopyOnly,
    MoveCounter,
    ObservableDel,
    ObservableMoveOnly,
    check_write_to,
)
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)


def test_unsafepointer_of_move_only_type():
    var actions = List[String]()
    var actions_ptr = LegacyUnsafePointer(to=actions).as_immutable()

    var ptr = LegacyUnsafePointer[
        mut=True, ObservableMoveOnly[actions_ptr.origin]
    ].alloc(1)
    ptr.init_pointee_move(ObservableMoveOnly(42, actions_ptr))
    assert_equal(len(actions_ptr[0]), 2)
    assert_equal(actions_ptr[0][0], "__init__")
    assert_equal(actions_ptr[0][1], "move ctor", msg="emplace_value")
    assert_equal(ptr[0].value, 42)

    # Stop compiler warnings
    var true = True

    if true:  # scope value
        var value = ptr.take_pointee()
        assert_equal(len(actions_ptr[0]), 3)
        assert_equal(actions_ptr[0][2], "move ctor")
        assert_equal(value.value, 42)

    ptr.free()
    assert_equal(len(actions_ptr[0]), 4)
    assert_equal(actions_ptr[0][3], "__del__")


def test_unsafepointer_move_pointee_move_count():
    var ptr = LegacyUnsafePointer[mut=True, MoveCounter[Int]].alloc(1)

    var value = MoveCounter(5)
    assert_equal(0, value.move_count)
    ptr.init_pointee_move(value^)

    # -----
    # Test that `LegacyUnsafePointer.move_pointee` performs exactly one move.
    # -----

    assert_equal(1, ptr[].move_count)

    var ptr_2 = LegacyUnsafePointer[mut=True, MoveCounter[Int]].alloc(1)
    ptr_2.init_pointee_move_from(ptr)

    assert_equal(2, ptr_2[].move_count)


def test_unsafepointer_init_pointee_copy():
    var ptr = LegacyUnsafePointer[mut=True, ExplicitCopyOnly].alloc(1)

    var orig = ExplicitCopyOnly(5)
    assert_equal(orig.copy_count, 0)

    # Test initialize pointee from `Copyable` type
    ptr.init_pointee_copy(orig)

    assert_equal(ptr[].value, 5)
    assert_equal(ptr[].copy_count, 1)


def test_refitem():
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(1)
    ptr[0] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)
    ptr.free()


def test_refitem_offset():
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(5)
    for i in range(5):
        ptr[i] = i
    for i in range(5):
        assert_equal(ptr[i], i)
    ptr.free()


def test_address_of():
    var local = 1
    assert_not_equal(0, Int(LegacyUnsafePointer[mut=True, Int](to=local)))
    _ = local


def test_pointer_to():
    var local = 1
    assert_not_equal(0, LegacyUnsafePointer(to=local)[])


def test_explicit_copy_of_pointer_address():
    var local = 1
    var ptr = LegacyUnsafePointer[mut=True, Int](to=local)
    var copy = LegacyUnsafePointer(other=ptr)
    assert_equal(Int(ptr), Int(copy))
    _ = local


def test_bitcast():
    var local = 1
    var ptr = LegacyUnsafePointer[mut=True, Int](to=local)
    var aliased_ptr = ptr.bitcast[SIMD[DType.uint8, 4]]()

    assert_equal(Int(ptr), Int(ptr.bitcast[Int]()))

    assert_equal(Int(ptr), Int(aliased_ptr))

    _ = local


def test_unsafepointer_string():
    var nullptr = LegacyUnsafePointer[mut=True, Int]()
    assert_equal(String(nullptr), "0x0")

    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(1)
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")
    ptr.free()


def test_eq():
    var local = 1
    # FIXME(#5133): should just be LegacyUnsafePointer[mut=False](to=local)
    var p1 = LegacyUnsafePointer(to=local).as_immutable()
    var p2 = p1
    assert_equal(Int(p1), Int(p2))

    var other_local = 2
    var p3 = LegacyUnsafePointer(to=other_local).as_immutable()
    assert_not_equal(Int(p1), Int(p3))

    var p4 = LegacyUnsafePointer(to=local).as_immutable()
    assert_equal(Int(p1), Int(p4))
    _ = local
    _ = other_local


def test_comparisons():
    var p1 = LegacyUnsafePointer[mut=True, Int].alloc(1)

    assert_true((p1 - 1) < p1)
    assert_true((p1 - 1) <= p1)
    assert_true(p1 <= p1)
    assert_true((p1 + 1) > p1)
    assert_true((p1 + 1) >= p1)
    assert_true(p1 >= p1)

    p1.free()


def test_unsafepointer_address_space():
    var p1 = LegacyUnsafePointer[
        mut=True, Int, address_space = AddressSpace(0)
    ].alloc(1)
    p1.free()

    var p2 = LegacyUnsafePointer[
        mut=True, Int, address_space = AddressSpace.GENERIC
    ].alloc(1)
    p2.free()


def test_unsafepointer_aligned_alloc():
    comptime alignment_1 = 32
    var ptr = LegacyUnsafePointer[mut=True, UInt8].alloc(
        1, alignment=alignment_1
    )
    var ptr_uint64 = UInt64(Int(ptr))
    ptr.free()
    assert_equal(ptr_uint64 % alignment_1, 0)

    comptime alignment_2 = 64
    var ptr_2 = LegacyUnsafePointer[mut=True, UInt8].alloc(
        1, alignment=alignment_2
    )
    var ptr_uint64_2 = UInt64(Int(ptr_2))
    ptr_2.free()
    assert_equal(ptr_uint64_2 % alignment_2, 0)

    comptime alignment_3 = 128
    var ptr_3 = LegacyUnsafePointer[mut=True, UInt8].alloc(
        1, alignment=alignment_3
    )
    var ptr_uint64_3 = UInt64(Int(ptr_3))
    ptr_3.free()
    assert_equal(ptr_uint64_3 % alignment_3, 0)


# Test that `LegacyUnsafePointer.alloc()` no longer artificially extends the lifetime
# of every local variable in methods where its used.
def test_unsafepointer_alloc_origin():
    # -----------------------------------------
    # Test with MutAnyOrigin alloc() origin
    # -----------------------------------------

    var did_del_1 = False

    # Allocate pointer with MutAnyOrigin.
    var ptr_1 = LegacyUnsafePointer[mut=True, Int].alloc(1).as_any_origin()

    var obj_1 = ObservableDel(LegacyUnsafePointer(to=did_del_1))

    # Object has not been deleted, because MutAnyOrigin is keeping it alive.
    assert_false(did_del_1)

    ptr_1.free()

    # Now that `ptr` is out of scope, `obj_1` was destroyed as well.
    assert_true(did_del_1)

    # ----------------------------------------
    # Test with default (empty) alloc() origin
    # ----------------------------------------

    var did_del_2 = False

    # Allocate pointer with empty origin.
    var ptr_2 = LegacyUnsafePointer[mut=True, Int].alloc(1)

    # Note: Set ObservableDel origin explicitly since it otherwise contains a
    #   MutAnyOrigin pointer that interferes with this test.
    _ = ObservableDel[origin_of(did_del_2)](LegacyUnsafePointer(to=did_del_2))

    # `obj_2` is ASAP destroyed, since `ptr_2` origin does not keep it alive.
    assert_true(did_del_2)

    ptr_2.free()


# NOTE: Tests fails due to a `LegacyUnsafePointer` size
# constraint failing to be satisfied.
#
# def test_unsafepointer_zero_size():
#     alias T = SIMD[DType.int32, 0]
#
#     var start_ptr = LegacyUnsafePointer[T].alloc(10)
#     var dest_ptr = start_ptr + 5
#
#     assert_true(start_ptr < dest_ptr)
#     assert_true(start_ptr != dest_ptr)


def test_indexing():
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(4)
    for i in range(4):
        ptr[i] = i

    assert_equal(ptr[Int(1)], 1)
    assert_equal(ptr[3], 3)

    ptr.free()


def test_indexing_simd():
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(4)
    for i in range(4):
        ptr[UInt8(i)] = i

    assert_equal(ptr[UInt8(1)], 1)
    assert_equal(ptr[UInt8(3)], 3)
    assert_equal(ptr[UInt16(1)], 1)
    assert_equal(ptr[UInt16(3)], 3)
    assert_equal(ptr[UInt32(1)], 1)
    assert_equal(ptr[UInt32(3)], 3)
    assert_equal(ptr[UInt64(1)], 1)
    assert_equal(ptr[UInt64(3)], 3)
    assert_equal(ptr[Int8(1)], 1)
    assert_equal(ptr[Int8(3)], 3)
    assert_equal(ptr[Int16(1)], 1)
    assert_equal(ptr[Int16(3)], 3)
    assert_equal(ptr[Int32(1)], 1)
    assert_equal(ptr[Int32(3)], 3)
    assert_equal(ptr[Int64(1)], 1)
    assert_equal(ptr[Int64(3)], 3)

    ptr.free()


def test_bool():
    var nullptr = LegacyUnsafePointer[mut=True, Int]()
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(1)

    assert_true(ptr.__bool__())
    assert_false(nullptr.__bool__())

    ptr.free()


def test_alignment():
    var ptr = LegacyUnsafePointer[mut=True, Int64].alloc(8, alignment=64)
    assert_equal(Int(ptr) % 64, 0)
    ptr.free()

    var ptr_2 = LegacyUnsafePointer[mut=True, UInt8].alloc(32, alignment=32)
    assert_equal(Int(ptr_2) % 32, 0)
    ptr_2.free()


def test_pointer_arithmetic():
    var ptr = LegacyUnsafePointer[mut=True, Int].alloc(5)
    for i in range(5):
        ptr[i] = i
    var x = UInt(3)
    var y = Int(4)
    assert_equal((ptr + x)[], 3)
    assert_equal((ptr + y)[], 4)

    var ptr2 = LegacyUnsafePointer[mut=True, Int].alloc(5)
    var ptr3 = ptr2
    ptr2 += UInt(3)
    assert_equal(ptr2, ptr3 + 3)
    ptr2 -= UInt(5)
    assert_equal(ptr2, ptr3 + (-2))
    assert_equal(ptr2 + UInt(1), ptr3 + (-1))
    assert_equal(ptr2 - UInt(4), ptr3 + (-6))

    ptr.free()
    ptr2.free()


def test_load_and_store_simd():
    var ptr = LegacyUnsafePointer[mut=True, Int8].alloc(16)
    for i in range(16):
        ptr[i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.load[width=4](i)
        assert_equal(
            vec,
            SIMD[DType.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )
    ptr.free()

    var ptr2 = LegacyUnsafePointer[mut=True, Int8].alloc(16)
    for i in range(0, 16, 4):
        ptr2.store(i, SIMD[DType.int8, 4](i))
    for i in range(16):
        assert_equal(ptr2[i], Int8(i // 4 * 4))
    ptr2.free()


def test_volatile_load_and_store_simd():
    var ptr = LegacyUnsafePointer[mut=True, Int8].alloc(16)
    for i in range(16):
        ptr[i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.load[width=4, volatile=True](i)
        assert_equal(
            vec,
            SIMD[DType.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )
    ptr.free()

    var ptr2 = LegacyUnsafePointer[mut=True, Int8].alloc(16)
    for i in range(0, 16, 4):
        ptr2.store[volatile=True](i, SIMD[DType.int8, 4](i))
    for i in range(16):
        assert_equal(ptr2[i], Int8(i // 4 * 4))
    ptr2.free()


# Test pointer merging with ternary operation.
def test_merge():
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    fn inner(cond: Bool, x: Int, mut a: List[Int], mut b: List[Int]):
        var either = LegacyUnsafePointer(to=a) if cond else LegacyUnsafePointer(
            to=b
        )
        either[].append(x)

    inner(True, 7, a, b)
    inner(False, 8, a, b)

    assert_equal(a, [1, 2, 3, 7])
    assert_equal(b, [4, 5, 6, 8])


def test_swap_pointees_trivial_move():
    var a = 42
    LegacyUnsafePointer(to=a).as_any_origin().swap_pointees(
        LegacyUnsafePointer(to=a).as_any_origin()
    )
    assert_equal(a, 42)

    var x = 1
    var y = 2
    LegacyUnsafePointer(to=x).swap_pointees(LegacyUnsafePointer(to=y))
    assert_equal(x, 2)
    assert_equal(y, 1)


def test_swap_pointees_non_trivial_move():
    var counter = MoveCounter[Int](42)
    LegacyUnsafePointer(to=counter).as_any_origin().swap_pointees(
        LegacyUnsafePointer(to=counter).as_any_origin()
    )
    # Pointers point to the same object, so no move should be performed
    assert_equal(counter.value, 42)
    assert_equal(counter.move_count, 0)

    var counterA = MoveCounter[Int](1)
    var counterB = MoveCounter[Int](2)
    LegacyUnsafePointer(to=counterA).swap_pointees(
        LegacyUnsafePointer(to=counterB)
    )

    assert_equal(counterA.value, 2)
    assert_equal(counterA.move_count, 1)

    assert_equal(counterB.value, 1)
    assert_equal(counterB.move_count, 2)


def test_as_any_origin_mutable():
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](
        LegacyUnsafePointer(to=deleted)
    )
    var x = 42

    var mutable = LegacyUnsafePointer(to=x).as_any_origin()
    assert_true(mutable.mut)
    assert_false(deleted)

    mutable[] = 55
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_any_origin_immutable():
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](
        LegacyUnsafePointer(to=deleted)
    )
    var x = 42

    var immutable = LegacyUnsafePointer(to=x).as_any_origin().as_immutable()
    assert_false(immutable.mut)
    assert_false(deleted)

    var _x = immutable[]
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_immutable():
    var x = 42
    var mutable = LegacyUnsafePointer(to=x)
    assert_true(mutable.mut)
    assert_false(mutable.as_immutable().mut)


def test_unsafe_mut_cast():
    var x = 42
    var ptr = LegacyUnsafePointer(to=x)
    var immutable = ptr.unsafe_mut_cast[False]()
    assert_false(immutable.mut)
    var _mutable = immutable.unsafe_mut_cast[True]()
    assert_true(_mutable.mut)


fn _ref_to[origin: ImmutOrigin](ref[origin] to: String):
    pass


def test_unsafe_origin_cast():
    var x = "hello"
    var y = "world"

    var ptr = LegacyUnsafePointer(to=x)
    _ref_to[origin_of(x)](ptr[])
    _ref_to[origin_of(y)](ptr.unsafe_origin_cast[origin_of(y)]()[])


def test_write_to():
    check_write_to(
        LegacyUnsafePointer[Int, origin=MutAnyOrigin](),
        expected="0x0",
        is_repr=False,
    )

    var x = 42
    check_write_to(LegacyUnsafePointer(to=x), contains="0x", is_repr=False)

    var s = String("hello")
    check_write_to(LegacyUnsafePointer(to=s), contains="0x", is_repr=False)


def test_write_repr_to():
    check_write_to(
        LegacyUnsafePointer[Int, origin=MutAnyOrigin](),
        expected=(
            "LegacyUnsafePointer[mut=True, Int,"
            " address_space=AddressSpace.GENERIC](0x0)"
        ),
        is_repr=True,
    )

    var x = 42
    check_write_to(
        LegacyUnsafePointer(to=x),
        contains=(
            "LegacyUnsafePointer[mut=True, Int,"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )

    var s = String("hello")
    check_write_to(
        LegacyUnsafePointer(to=s),
        contains=(
            "LegacyUnsafePointer[mut=True, String,"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
