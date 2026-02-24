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

from compile import compile_info
from memory import UnsafePointer, alloc
from sys import size_of

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

# ---------------------------------------------------------------------------- #
# New tests for `UnsafePointer`
# ---------------------------------------------------------------------------- #


def _mutable_pointer(p: MutUnsafePointer[Int, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def _immutable_pointer(p: ImmutUnsafePointer[Int, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def _mutable_any_pointer(p: UnsafePointer[Int, MutAnyOrigin, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def _immutable_any_pointer(p: UnsafePointer[Int, ImmutAnyOrigin, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def _parameterized_pointer(p: UnsafePointer[Int, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def _named_origin[
    mut: Bool, //, origin: Origin[mut=mut]
](p: UnsafePointer[Int, origin, ...]):
    assert_true(p)
    assert_equal(p[], 42)


def test_mutable_conversions():
    var x = 42
    var p = UnsafePointer(to=x)
    _named_origin[origin_of(x)](p)
    _mutable_pointer(p)
    _immutable_pointer(p)
    _mutable_any_pointer(p)
    _immutable_any_pointer(p)
    _parameterized_pointer(p)


def test_immutable_conversions():
    var x = 42
    var p = UnsafePointer(to=x).as_immutable()
    _named_origin[mut=False, origin_of(x)](p)
    _immutable_pointer(p)
    _immutable_any_pointer(p)
    _parameterized_pointer(p)


def test_mutable_any_conversions():
    var x = 42
    var p = UnsafePointer(to=x).as_any_origin()
    _mutable_pointer(p)
    _immutable_pointer(p)
    _mutable_any_pointer(p)
    _immutable_any_pointer(p)
    _parameterized_pointer(p)


def test_immutable_any_conversions():
    var x = 42
    var p = UnsafePointer(to=x).as_immutable().as_any_origin()
    _immutable_pointer(p)
    _immutable_any_pointer(p)
    _parameterized_pointer(p)


# ---------------------------------------------------------------------------- #
# Copied tests from `test_unsafepointer.mojo`
# ---------------------------------------------------------------------------- #


def test_unsafepointer_of_move_only_type():
    var actions = List[String]()
    var actions_ptr = UnsafePointer(to=actions).as_immutable()

    comptime ObserveType = ObservableMoveOnly[actions_ptr.origin]

    var ptr = alloc[ObserveType](1)
    ptr.init_pointee_move(ObserveType(42, actions_ptr))
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
    var ptr = alloc[MoveCounter[Int]](1)

    var value = MoveCounter(5)
    assert_equal(0, value.move_count)
    ptr.init_pointee_move(value^)

    # -----
    # Test that `UnsafePointer.move_pointee` performs exactly one move.
    # -----

    assert_equal(1, ptr[].move_count)

    var ptr_2 = alloc[MoveCounter[Int]](1)
    ptr_2.init_pointee_move_from(ptr)

    assert_equal(2, ptr_2[].move_count)


def test_unsafepointer_init_pointee_copy():
    var ptr = alloc[ExplicitCopyOnly](1)

    var orig = ExplicitCopyOnly(5)
    assert_equal(orig.copy_count, 0)

    # Test initialize pointee from `Copyable` type
    ptr.init_pointee_copy(orig)

    assert_equal(ptr[].value, 5)
    assert_equal(ptr[].copy_count, 1)


def test_refitem():
    var ptr = alloc[Int](1)
    ptr[0] = 0
    ptr[] += 1
    assert_equal(ptr[], 1)
    ptr.free()


def test_refitem_offset():
    var ptr = alloc[Int](5)
    for i in range(5):
        ptr[i] = i
    for i in range(5):
        assert_equal(ptr[i], i)
    ptr.free()


def test_address_of():
    var local = 1
    assert_not_equal(0, Int(UnsafePointer[Int](to=local)))
    _ = local


def test_pointer_to():
    var local = 1
    assert_not_equal(0, UnsafePointer(to=local)[])


def test_explicit_copy_of_pointer_address():
    var local = 1
    var ptr = UnsafePointer[Int](to=local)
    var copy = UnsafePointer.copy(ptr)
    assert_equal(Int(ptr), Int(copy))
    _ = local


def test_bitcast():
    var local = 1
    var ptr = UnsafePointer[Int](to=local)
    var aliased_ptr = ptr.bitcast[SIMD[DType.uint8, 4]]()

    assert_equal(Int(ptr), Int(ptr.bitcast[Int]()))

    assert_equal(Int(ptr), Int(aliased_ptr))

    _ = local


def test_unsafepointer_string():
    var nullptr = UnsafePointer[Int, MutExternalOrigin]()
    assert_equal(String(nullptr), "0x0")

    var ptr = alloc[Int](1)
    assert_true(String(ptr).startswith("0x"))
    assert_not_equal(String(ptr), "0x0")
    ptr.free()


def test_eq():
    var local = 1
    # FIXME(#5133): should just be UnsafePointer[mut=False](to=local)
    var p1 = UnsafePointer(to=local).as_immutable()
    var p2 = p1
    assert_equal(p1, p2)

    var other_local = 2
    var p3 = UnsafePointer(to=other_local).as_immutable()
    assert_not_equal(Int(p1), Int(p3))

    var p4 = UnsafePointer(to=local).as_immutable()
    assert_equal(p1, p4)
    _ = local
    _ = other_local


def test_comparisons():
    var p1 = alloc[Int](1)

    assert_true((p1 - 1) < p1)
    assert_true((p1 - 1) <= p1)
    assert_true(p1 <= p1)
    assert_true((p1 + 1) > p1)
    assert_true((p1 + 1) >= p1)
    assert_true(p1 >= p1)

    p1.free()


def test_unsafepointer_address_space():
    var p1 = alloc[Int](1).address_space_cast[AddressSpace(0)]()
    p1.free()

    var p2 = alloc[Int](1).address_space_cast[AddressSpace.GENERIC]()
    p2.free()


def test_unsafepointer_aligned_alloc():
    comptime alignment_1 = 32
    var ptr = alloc[UInt8](1, alignment=alignment_1)
    var ptr_uint64 = UInt64(Int(ptr))
    ptr.free()
    assert_equal(ptr_uint64 % alignment_1, 0)

    comptime alignment_2 = 64
    var ptr_2 = alloc[UInt8](1, alignment=alignment_2)
    var ptr_uint64_2 = UInt64(Int(ptr_2))
    ptr_2.free()
    assert_equal(ptr_uint64_2 % alignment_2, 0)

    comptime alignment_3 = 128
    var ptr_3 = alloc[UInt8](1, alignment=alignment_3)
    var ptr_uint64_3 = UInt64(Int(ptr_3))
    ptr_3.free()
    assert_equal(ptr_uint64_3 % alignment_3, 0)


# Test that `alloc` no longer artificially extends the lifetime
# of every local variable in methods where its used.
def test_unsafepointer_alloc_origin():
    # -----------------------------------------
    # Test with MutAnyOrigin alloc() origin
    # -----------------------------------------

    var did_del_1 = False

    # Allocate pointer with MutAnyOrigin.
    var ptr_1 = alloc[Int](1).as_any_origin()

    var obj_1 = ObservableDel(UnsafePointer(to=did_del_1))

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
    var ptr_2 = alloc[Int](1)

    # Note: Set ObservableDel origin explicitly since it otherwise contains a
    #   MutAnyOrigin pointer that interferes with this test.
    _ = ObservableDel[origin_of(did_del_2)](UnsafePointer(to=did_del_2))

    # `obj_2` is ASAP destroyed, since `ptr_2` origin does not keep it alive.
    assert_true(did_del_2)

    ptr_2.free()


# NOTE: Tests fails due to a `UnsafePointer` size
# constraint failing to be satisfied.
#
# def test_unsafepointer_zero_size():
#     alias T = SIMD[DType.int32, 0]
#
#     var start_ptr = UnsafePointer[T].alloc(10)
#     var dest_ptr = start_ptr + 5
#
#     assert_true(start_ptr < dest_ptr)
#     assert_true(start_ptr != dest_ptr)


def test_indexing():
    var ptr = alloc[Int](4)
    for i in range(4):
        ptr[i] = i

    assert_equal(ptr[Int(1)], 1)
    assert_equal(ptr[3], 3)

    ptr.free()


def test_indexing_simd():
    var ptr = alloc[Int](4)
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
    var nullptr = UnsafePointer[Int, MutExternalOrigin]()
    var ptr = alloc[Int](1)

    assert_true(ptr.__bool__())
    assert_false(nullptr.__bool__())

    ptr.free()


def test_alignment():
    var ptr = alloc[Int64](8, alignment=64)
    assert_equal(Int(ptr) % 64, 0)
    ptr.free()

    var ptr_2 = alloc[UInt8](32, alignment=32)
    assert_equal(Int(ptr_2) % 32, 0)
    ptr_2.free()


def test_offset():
    var ptr = alloc[Int](5)
    for i in range(5):
        ptr[i] = i
    var x = UInt(3)
    var y = Int(4)
    assert_equal((ptr + x)[], 3)
    assert_equal((ptr + y)[], 4)

    var ptr2 = alloc[Int](5)
    var ptr3 = ptr2
    ptr2 += UInt(3)
    assert_equal(ptr2, ptr3 + 3)
    ptr2 -= UInt(5)
    assert_equal(ptr2, ptr3 + -2)
    assert_equal(ptr2 + UInt(1), ptr3 + -1)
    assert_equal(ptr2 - UInt(4), ptr3 + -6)

    ptr.free()
    ptr2.free()


def test_load_and_store_simd():
    var ptr = alloc[Int8](16)
    for i in range(16):
        ptr[i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.load[width=4](i)
        assert_equal(
            vec,
            SIMD[DType.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )
    ptr.free()

    var ptr2 = alloc[Int8](16)
    for i in range(0, 16, 4):
        ptr2.store(i, SIMD[DType.int8, 4](i))
    for i in range(16):
        assert_equal(ptr2[i], Int8(i // 4 * 4))
    ptr2.free()


def test_load_and_store_simd_bool():
    # Regression test: storing SIMD[DType.bool, N] with width > 1 then
    # loading element-wise should give correct results (github.com/modular/modular/issues/5875).
    var p = alloc[Scalar[DType.bool]](4)

    p.store(0, SIMD[DType.bool, 2](True, False))
    assert_true(p[0])
    assert_false(p[1])
    for i in range(2):
        assert_equal(p.load[width=2](0)[i], p[i])

    p.store(0, SIMD[DType.bool, 4](False, True, True, False))
    assert_false(p[0])
    assert_true(p[1])
    assert_true(p[2])
    assert_false(p[3])
    for i in range(4):
        assert_equal(p.load[width=4](0)[i], p[i])

    p.free()


def test_volatile_load_and_store_simd():
    var ptr = alloc[Int8](16)
    for i in range(16):
        ptr[i] = Int8(i)
    for i in range(0, 16, 4):
        var vec = ptr.load[width=4, volatile=True](i)
        assert_equal(
            vec,
            SIMD[DType.int8, 4](Int8(i), Int8(i + 1), Int8(i + 2), Int8(i + 3)),
        )
    ptr.free()

    var ptr2 = alloc[Int8](16)
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
        var either = UnsafePointer(to=a) if cond else UnsafePointer(to=b)
        either[].append(x)

    inner(True, 7, a, b)
    inner(False, 8, a, b)

    assert_equal(a, [1, 2, 3, 7])
    assert_equal(b, [4, 5, 6, 8])


def test_swap_pointees_trivial_move():
    var a = 42
    UnsafePointer(to=a).as_any_origin().swap_pointees(
        UnsafePointer(to=a).as_any_origin()
    )
    assert_equal(a, 42)

    var x = 1
    var y = 2
    UnsafePointer(to=x).swap_pointees(UnsafePointer(to=y))
    assert_equal(x, 2)
    assert_equal(y, 1)


def test_swap_pointees_non_trivial_move():
    var counter = MoveCounter[Int](42)
    UnsafePointer(to=counter).as_any_origin().swap_pointees(
        UnsafePointer(to=counter).as_any_origin()
    )
    # Pointers point to the same object, so no move should be performed
    assert_equal(counter.value, 42)
    assert_equal(counter.move_count, 0)

    var counterA = MoveCounter[Int](1)
    var counterB = MoveCounter[Int](2)
    UnsafePointer(to=counterA).swap_pointees(UnsafePointer(to=counterB))

    assert_equal(counterA.value, 2)
    assert_equal(counterA.move_count, 1)

    assert_equal(counterB.value, 1)
    assert_equal(counterB.move_count, 2)


def test_as_any_origin_mutable():
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](UnsafePointer(to=deleted))
    var x = 42

    var mutable = UnsafePointer(to=x).as_any_origin()
    assert_true(mutable.mut)
    assert_false(deleted)

    mutable[] = 55
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_any_origin_immutable():
    var deleted = False
    var observer = ObservableDel[origin_of(deleted)](UnsafePointer(to=deleted))
    var x = 42

    var immutable = UnsafePointer(to=x).as_any_origin().as_immutable()
    assert_false(immutable.mut)
    assert_false(deleted)

    var _x = immutable[]
    assert_true(deleted)  # AnyOrigin extends all lifetimes


def test_as_immutable():
    var x = 42
    var mutable = UnsafePointer(to=x)
    assert_true(mutable.mut)
    assert_false(mutable.as_immutable().mut)


def test_unsafe_mut_cast():
    var x = 42
    var ptr = UnsafePointer(to=x)
    var immutable = ptr.unsafe_mut_cast[False]()
    assert_false(immutable.mut)
    var _mutable = immutable.unsafe_mut_cast[True]()
    assert_true(_mutable.mut)


fn _ref_to[origin: ImmutOrigin](ref[origin] to: String):
    pass


def test_unsafe_origin_cast():
    var x = "hello"
    var y = "world"

    var ptr = UnsafePointer(to=x)
    _ref_to[origin_of(x)](ptr[])
    _ref_to[origin_of(y)](ptr.unsafe_origin_cast[origin_of(y)]()[])


fn _ptr_to_int(ptr: UnsafePointer[Int, MutExternalOrigin]) -> Int:
    return Int(ptr)


def test_ptr_to_int_llvm_lowering():
    var info = compile_info[_ptr_to_int, emission_kind="llvm-opt"]()
    # https://llvm.org/docs/LangRef.html#ptrtoint-to-instruction
    # We need to check `ptrtoint` is used instead of `ptrtoaddr` to ensure
    # pointer provenance is preserved for the default ptr -> int conversion.
    assert_true("ptrtoint" in info.asm)
    assert_false("ptrtoaddr" in info.asm)


fn _from_address(x: Int, out result: UnsafePointer[Int, MutExternalOrigin]):
    result = type_of(result)(unsafe_from_address=x)


def test_unsafe_from_address_llvm_lowering():
    var info = compile_info[_from_address, emission_kind="llvm-opt"]()
    assert_true("inttoptr" in info.asm)


def test_unsafe_from_address():
    var x = 42
    var ptr = UnsafePointer(to=x)
    var ptr2 = type_of(ptr)(unsafe_from_address=Int(ptr))
    assert_equal(ptr2[], 42)

    var ptr3 = UnsafePointer[Int, MutExternalOrigin](unsafe_from_address=42)
    assert_true(ptr3)


def test_write_to():
    check_write_to(
        UnsafePointer[Int, MutAnyOrigin](), expected="0x0", is_repr=False
    )

    var x = 42
    check_write_to(UnsafePointer(to=x), contains="0x", is_repr=False)

    var s = String("hello")
    check_write_to(UnsafePointer(to=s), contains="0x", is_repr=False)


def test_write_repr_to():
    check_write_to(
        UnsafePointer[Int, MutAnyOrigin](),
        expected=(
            "UnsafePointer[mut=True, Int,"
            " address_space=AddressSpace.GENERIC](0x0)"
        ),
        is_repr=True,
    )

    var x = 42
    check_write_to(
        UnsafePointer(to=x),
        contains=(
            "UnsafePointer[mut=True, Int,"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        UnsafePointer(to=x).as_immutable(),
        contains=(
            "UnsafePointer[mut=False, Int,"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )
    check_write_to(
        UnsafePointer(to=x).address_space_cast[AddressSpace.SHARED](),
        contains=(
            "UnsafePointer[mut=True, Int, address_space=AddressSpace.SHARED](0x"
        ),
        is_repr=True,
    )

    var s = String("hello")
    check_write_to(
        UnsafePointer(to=s),
        contains=(
            "UnsafePointer[mut=True, String,"
            " address_space=AddressSpace.GENERIC](0x"
        ),
        is_repr=True,
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
