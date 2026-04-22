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

from std.ffi import _Global
from std.memory import UnsafeMaybeUninit
from std.os import abort
from std.sys import size_of

from test_utils import (
    MoveCopyCounter,
    ObservableDel,
    ConfigureTrivial,
    MoveOnly,
    ExplicitDelOnly,
    NonMovable,
    Observable,
    check_write_to,
)
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from std.benchmark import keep

from std.utils import Variant
from std.utils._nicheable import UnsafeNicheable

comptime TEST_VARIANT_POISON = _Global[
    "TEST_VARIANT_POISON", _initialize_poison
]


def _initialize_poison() -> Bool:
    return False


def _poison_ptr() -> UnsafePointer[Bool, MutExternalOrigin]:
    try:
        return TEST_VARIANT_POISON.get_or_create_ptr()
    except:
        abort("Failed to get or create TEST_VARIANT_POISON")


def assert_no_poison() raises:
    assert_false(_poison_ptr().take_pointee())


struct Poison(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        _poison_ptr().init_pointee_move(True)

    def __init__(out self, *, deinit take: Self):
        _poison_ptr().init_pointee_move(True)

    def __del__(deinit self):
        _poison_ptr().init_pointee_move(True)


comptime TestVariant = Variant[MoveCopyCounter, Poison]


def test_basic() raises:
    comptime IntOrString = Variant[Int, String]
    var i = IntOrString(4)
    var s = IntOrString("4")

    # isa
    assert_true(i.isa[Int]())
    assert_false(i.isa[String]())
    assert_true(s.isa[String]())
    assert_false(s.isa[Int]())

    # get
    assert_equal(4, i[Int])
    assert_equal("4", s[String])
    # we don't test what happens when you `get` the wrong type.
    # have fun!

    # set
    i.set[String]("i")
    assert_false(i.isa[Int]())
    assert_true(i.isa[String]())
    assert_equal("i", i[String])


def test_copy() raises:
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call copyinit
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)
    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_explicit_copy() raises:
    var v1 = TestVariant(MoveCopyCounter())

    # Perform explicit copy
    var v2 = v1.copy()

    # Test copy counts
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)

    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_move() raises:
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call moveinit
    assert_equal(v1[MoveCopyCounter].moved, 1)
    assert_equal(v2[MoveCopyCounter].moved, 1)
    # test that we didn't call the other moveinit too!
    assert_no_poison()


def test_del() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    _ = v1^  # call __del__
    assert_true(deleted)
    # test that we didn't call the other deleter too!
    assert_no_poison()


def test_set_calls_deleter() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var deleted2: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    v1.set(ObservableDel(UnsafePointer(to=deleted2)))
    assert_true(deleted)
    assert_false(deleted2)
    _ = v1^
    assert_true(deleted2)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_replace() raises:
    var v1: Variant[Int, String] = 998
    var x = v1.replace[String, Int]("hello")

    assert_equal(x, 998)


def test_take_doesnt_call_deleter() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    assert_false(deleted)
    var v2 = v1^.unsafe_take[ObservableDel[]]()
    assert_false(deleted)
    _ = v2
    assert_true(deleted)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_get_returns_mutable_reference() raises:
    var v1: Variant[Int, String] = 42
    var x = v1[Int]
    assert_equal(42, x)
    x = 100
    assert_equal(100, x)
    v1.set[String]("hello")
    assert_equal(100, x)  # the x reference is still valid

    var v2: Variant[Int, String] = "something"
    v2[String] = "something else"
    assert_equal(v2[String], "something else")


def test_is_type_supported() raises:
    var _x: Variant[Float64, Int32]
    assert_equal(_x.is_type_supported[Float64](), True)
    assert_equal(_x.is_type_supported[Int32](), True)
    assert_equal(_x.is_type_supported[Float32](), False)
    assert_equal(_x.is_type_supported[UInt32](), False)
    var _y: Variant[SIMD[DType.uint8, 2], SIMD[DType.uint8, 4]]
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 2]](), True)
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 4]](), True)
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 8]](), False)


def test_variant_works_with_move_only_types() raises:
    var v1 = Variant[MoveOnly[Int], MoveOnly[String]](MoveOnly[Int](42))
    var v2 = v1^
    assert_equal(v2[MoveOnly[Int]].data, 42)


def test_variant_linear_type_take() raises:
    var v = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))

    var x = v^.take[ExplicitDelOnly]()

    var data = x.data
    # Destroy before potentially raising after assert
    x^.destroy()
    assert_equal(data, 5)


def test_variant_linear_type_destroy_with() raises:
    # Test destroying a linear variant element in-place
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    v1^.destroy_with(ExplicitDelOnly.destroy)

    # Test destroying a non-linear variant element in-place
    var v2 = Variant[ExplicitDelOnly, String]("notlinear")
    v2^.destroy_with(String.__del__)


def test_variant_linear_type_move() raises:
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    var v2 = v1^

    v2^.destroy_with(ExplicitDelOnly.destroy)


def test_variant_trivial_del() raises:
    comptime yes = ConfigureTrivial[del_is_trivial=True]
    comptime no = ConfigureTrivial[del_is_trivial=False]

    assert_true(Variant[yes].__del__is_trivial)
    assert_false(Variant[no].__del__is_trivial)
    assert_false(Variant[yes, no].__del__is_trivial)

    # TODO (MOCO-3016):
    # check variant of linear type
    # assert_false(Variant[LinearType].__del__is_trivial)


def test_variant_trivial_copyinit() raises:
    comptime yes = ConfigureTrivial[copyinit_is_trivial=True]
    comptime no = ConfigureTrivial[copyinit_is_trivial=False]

    assert_true(Variant[yes].__copy_ctor_is_trivial)
    assert_false(Variant[no].__copy_ctor_is_trivial)
    assert_false(Variant[yes, no].__copy_ctor_is_trivial)

    # check variant of move-only type
    assert_false(Variant[MoveOnly[Int]].__copy_ctor_is_trivial)


def test_variant_trivial_moveinit() raises:
    comptime yes = ConfigureTrivial[moveinit_is_trivial=True]
    comptime no = ConfigureTrivial[moveinit_is_trivial=False]

    assert_true(Variant[yes].__move_ctor_is_trivial)
    assert_false(Variant[no].__move_ctor_is_trivial)
    assert_false(Variant[yes, no].__move_ctor_is_trivial)

    # check variant of non-movable type
    # # TODO(MOCO-3383): Compiler issue with folding non-struct types
    # assert_false(Variant[NonMovable].__move_ctor_is_trivial)


def test_variant_write_to() raises:
    var v = Variant[Int, String](42)
    check_write_to(v, expected="42", is_repr=False)
    v = "hello"
    check_write_to(v, expected="hello", is_repr=False)


def test_variant_write_repr_to() raises:
    var v = Variant[Int, String](42)
    check_write_to(v, expected="Variant[Int, String](Int(42))", is_repr=True)
    v = "hello"
    check_write_to(v, expected="Variant[Int, String]('hello')", is_repr=True)


@fieldwise_init
struct EmptyAndTrivial[Tag: Int = 0](TrivialRegisterPassable):
    pass


def test_variant_niche_optimization_size() raises:
    comptime NicheableType = Observable[opt_into_unsafe_niche=True]

    # Fits the optional-niche criteria
    assert_equal(
        size_of[Variant[NicheableType, EmptyAndTrivial[]]](),
        size_of[NicheableType](),
    )
    assert_equal(
        size_of[Variant[EmptyAndTrivial[], NicheableType]](),
        size_of[NicheableType](),
    )

    # Int does _not_ implement `UnsafeNicheable`
    assert_true(size_of[Variant[Int, EmptyAndTrivial[]]]() > size_of[Int]())
    assert_true(size_of[Variant[EmptyAndTrivial[], Int]]() > size_of[Int]())

    # More than 1 "empty type" does not opt into the niche optimization
    assert_true(
        size_of[
            Variant[NicheableType, EmptyAndTrivial[0], EmptyAndTrivial[1]]
        ]()
        > size_of[NicheableType]()
    )


def test_niched_variant_correctly_handles_lifecycle() raises:
    var copies = 0
    var moves = 0
    var dels = 0
    comptime NicheableType = Observable[
        CopyOrigin=origin_of(copies),
        MoveOrigin=origin_of(moves),
        DelOrigin=origin_of(dels),
        opt_into_unsafe_niche=True,
    ]
    comptime VariantType = Variant[NicheableType, EmptyAndTrivial[]]

    var empty = VariantType(EmptyAndTrivial())
    _ = empty^
    assert_equal(copies, 0)
    assert_equal(moves, 0)
    assert_equal(dels, 0)

    var observe = VariantType(
        NicheableType(
            copies=Pointer(to=copies),
            moves=Pointer(to=moves),
            dels=Pointer(to=dels),
        )
    )
    assert_equal(copies, 0)
    assert_equal(moves, 1)
    assert_equal(dels, 0)

    var copy_observe = observe.copy()
    assert_equal(copies, 1)
    assert_equal(moves, 2)
    assert_equal(dels, 0)

    _ = copy_observe^
    _ = observe^
    assert_equal(copies, 1)
    assert_equal(moves, 2)
    assert_equal(dels, 2)


def test_variant_eq() raises:
    comptime IntOrStr = Variant[Int, String]
    assert_true(IntOrStr(42) == IntOrStr(42))
    assert_true(IntOrStr("hello") == IntOrStr("hello"))
    assert_false(IntOrStr(42) == IntOrStr(99))
    assert_false(IntOrStr("hello") == IntOrStr("world"))
    assert_false(IntOrStr(42) == IntOrStr("42"))

    assert_true(IntOrStr(1) != IntOrStr(2))
    assert_true(IntOrStr(42) != IntOrStr("42"))
    assert_false(IntOrStr(1) != IntOrStr(1))

    comptime V3 = Variant[Int, String, Float64]
    assert_true(V3(42) == V3(42))
    assert_false(V3(42) == V3(3.14))
    assert_false(V3(42) == V3("42"))

    comptime V1 = Variant[Int]
    assert_true(V1(42) == V1(42))
    assert_false(V1(42) != V1(42))
    assert_true(V1(42) != V1(99))


def test_variant_hash() raises:
    comptime IntOrStr = Variant[Int, String]
    assert_equal(hash(IntOrStr(42)), hash(IntOrStr(42)))
    assert_equal(hash(IntOrStr("hello")), hash(IntOrStr("hello")))
    assert_true(hash(IntOrStr(42)) != hash(IntOrStr(99)))
    assert_true(hash(IntOrStr(42)) != hash(IntOrStr("42")))

    comptime V1 = Variant[Int]
    assert_equal(hash(V1(42)), hash(V1(42)))
    assert_true(hash(V1(42)) != hash(V1(99)))


def test_variant_conditional_conformances() raises:
    assert_true(conforms_to(Variant[Int, String], Equatable))
    assert_true(conforms_to(Variant[Int], Equatable))
    assert_true(conforms_to(Variant[Int, String], Hashable))
    assert_true(conforms_to(Variant[Int], Hashable))
    assert_true(conforms_to(Variant[Int, String], Writable))
    assert_true(conforms_to(Variant[Int], Writable))

    assert_false(conforms_to(Variant[MoveOnly[Int]], Equatable))
    assert_false(conforms_to(Variant[MoveOnly[Int]], Hashable))
    assert_false(conforms_to(Variant[MoveOnly[Int]], Writable))

    # Copyable: all types Copyable
    assert_true(conforms_to(Variant[Int, String], Copyable))
    assert_true(conforms_to(Variant[Int], Copyable))

    # Copyable: move-only type
    assert_false(conforms_to(Variant[MoveOnly[Int]], Copyable))
    assert_false(conforms_to(Variant[Int, MoveOnly[Int]], Copyable))

    # ImplicitlyCopyable: all types ImplicitlyCopyable
    assert_true(conforms_to(Variant[Int, Bool], ImplicitlyCopyable))

    # ImplicitlyCopyable: not all types ImplicitlyCopyable
    assert_false(conforms_to(Variant[MoveOnly[Int]], ImplicitlyCopyable))

    # RegisterPassable: all types RP
    assert_true(conforms_to(Variant[Int, Bool], RegisterPassable))
    assert_true(conforms_to(Variant[Int], RegisterPassable))

    # RegisterPassable: all types non-RP
    assert_false(conforms_to(Variant[String, List[Int]], RegisterPassable))

    # RegisterPassable: mixture of RP and non-RP
    assert_false(conforms_to(Variant[Int, String], RegisterPassable))
    assert_false(conforms_to(Variant[Bool, List[Int], Int], RegisterPassable))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
