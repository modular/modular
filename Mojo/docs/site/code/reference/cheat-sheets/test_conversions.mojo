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
# test_conversions.mojo
# Tests for the Mojo "conversions" cheat-sheet card.
#
# Exercises the make / convert / access casts shown on the card so a claim that
# drifts stops compiling or fails an assert.
#
# Not tested (need a GPU/Python, or no portable runtime API to assert):
#   - .value()/.take() on an empty Optional abort (can't assert an abort)
#   - @implicit conversions (a compile-time behavior, not a runtime value)
#   - conversions that don't compile (Int16(s), Float32 from a Floatable,
#     mixed-width arithmetic): compile errors can't be asserted at runtime
#   - Python conversions (need the Python runtime); Float8 (needs a GPU)
from std.ffi import external_call
from std.memory import bitcast, dealloc
from std.reflection import reflect
from std.sys.info import size_of
from std.testing import assert_equal, assert_true, assert_false, assert_raises


# A user type joins T(x) conversion by conforming to the conversion traits.
@fieldwise_init
struct Celsius(Boolable, Floatable, Intable, Writable):
    var deg: Float64

    def __float__(self) -> Float64:
        return self.deg

    def __int__(self) -> Int:
        return Int(self.deg)

    def __bool__(self) -> Bool:
        return self.deg != 0

    def write_to(self, mut w: Some[Writer]):
        w.write(self.deg, "C")


def test_establishing_types() raises:
    # explicit typing from literal expressions
    assert_equal(Int(32), 32)
    assert_equal(Int16(32), 32)
    # new forms across types
    var f: Float64 = 3.9
    var i: Int = 7
    assert_equal(Int64(f), 3)
    assert_equal(Float64(i), 7.0)
    assert_equal(String(i), "7")
    # the trait table: Intable reaches any integer type, Floatable only Float64
    var c = Celsius(21.5)
    assert_equal(Int16(c), 21)
    assert_equal(UInt8(c), 21)
    assert_equal(Float64(c), 21.5)
    assert_true(Bool(c))
    assert_equal(String(c), "21.5C")
    # strings convert only to Int and Float64; go through one, then narrow
    var s = String("12")
    assert_equal(Int(s), 12)
    assert_equal(Float64(s), 12.0)
    assert_equal(Int16(Int(s)), 12)
    with assert_raises():
        _ = Int(String("two"))


def test_casting() raises:
    # same-width signed/unsigned keeps the bits
    assert_equal(UInt(Int(-1)), UInt.MAX)
    var f64: Float64 = 1.0
    var f32: Float32 = 1.0
    var i32: Int32 = 2
    assert_equal(f64.cast[.int32](), 1)  # convert the value
    assert_equal(bitcast[.uint32](f32), 1065353216)  # reinterpret the bits
    assert_equal(f64 + Float64(i32), 3.0)  # explicit, never implicit
    assert_equal(Float64(f32) + f64, 2.0)  # widen one side to match
    # widening preserves precision; narrowing may lose it
    var tenth: Float32 = 0.1
    assert_equal(Float64(tenth).cast[.float32](), tenth)
    assert_true(Float64(Float64(0.1).cast[.float32]()) != 0.1)


def test_numbers() raises:
    # make an Int: from Intable, a scalar, a String (raising)
    assert_equal(Int(True), 1)
    assert_equal(Int(Int32(5)), 5)
    assert_equal(Int("42"), 42)
    # Int is a SIMD scalar, so .cast works
    assert_equal(Int(5).cast[.int8](), Int8(5))
    # Int <-> UInt share bits: the reinterpretation round-trips
    assert_equal(Int(UInt(Int(-1))), -1)
    # make a Float64; Float -> Int truncates toward zero
    assert_equal(Float64(3), 3.0)
    assert_equal(Float64("1.5"), 1.5)
    assert_equal(Int(Float64(3.9)), 3)
    # Bool: from Boolable, from None
    assert_true(Bool(1))
    assert_true(Bool("x"))
    assert_false(Bool(""))
    assert_false(Bool(None))
    # number -> Bool / Float / String
    assert_equal(Int(True), 1)
    assert_equal(Int(False), 0)
    assert_equal(Float64(True), 1.0)
    assert_equal(String(True), "True")


def test_numeric_creation() raises:
    var int_default = 42  # integer literals default to Int
    var float_default = 3.14  # floating-point literals default to Float64
    assert_equal(
        String(reflect[type_of(int_default)].name()),
        String(reflect[Int].name()),
    )
    assert_equal(
        String(reflect[type_of(float_default)].name()),
        String(reflect[Float64].name()),
    )
    assert_true(int_default == 42 and float_default > 3.0)
    assert_true(Float16(3.14) > 3.1)  # a literal takes the type you name
    assert_equal(size_of[Float64](), 8)  # Float64 is always 64 bits
    assert_equal(size_of[Int](), size_of[OpaquePointer[MutUntrackedOrigin]]())
    var i32 = Int32(42)  # a literal becomes the type you name
    var f32 = Float32(3.14)
    assert_equal(i32, 42)
    assert_true(f32 > 3.13 and f32 < 3.15)
    var c = Celsius(21.5)
    assert_equal(Int32(c), 21)  # any Intable, to any integer dtype
    assert_equal(Float64(c), 21.5)  # any Floatable; Float64 only
    with assert_raises():
        _ = Int("two")
    with assert_raises():
        _ = Float64("two")


def test_numeric_conversion() raises:
    var fp64: Float64 = 3.9
    assert_equal(fp64.cast[DType.int32](), 3)  # .cast works on Float64
    assert_equal(Int(300).cast[DType.int8](), 44)  # integer casts wrap
    assert_equal(Float64(-2.5).cast[DType.int32](), -2)  # toward zero
    assert_equal(UInt(Int(-1)), UInt.MAX)  # same bits
    assert_equal(Float64(7), 7.0)
    assert_equal(String(42), "42")
    assert_true(Bool(3))
    assert_false(Bool(0.0))
    var narrow = Float64(0.1).cast[DType.float32]()
    assert_true(Float64(narrow) != 0.1)  # precision loss


def test_simd() raises:
    var v = SIMD[.int32, 4](7)  # splat one value to all lanes
    assert_equal(v.reduce_add(), 28)
    var v2 = SIMD[.int32, 4](1, 2, 3, 4)  # per-lane (N matches arg count)
    assert_equal(v2[2], 3)
    assert_equal(v2.cast[.float64]()[0], 1.0)  # new dtype, same lane count
    var s: Int32 = 9
    assert_equal(SIMD[.int32, 4](s).reduce_add(), 36)  # scalar -> splat


def test_string() raises:
    # make from any Writable
    assert_equal(String(42), "42")
    assert_equal(String(True), "True")
    # convert: parse
    assert_equal(Int("7"), 7)
    assert_equal(Float64("2.5"), 2.5)
    assert_true(Bool("x"))
    assert_false(Bool(String("")))
    # access: byte/codepoint index + slice; grapheme single-index only
    var s = String("abcde")
    assert_equal(s[byte=0], "a")
    assert_equal(s[byte=1:3], "bc")
    assert_equal(s[codepoint=1], "b")
    assert_equal(s[codepoint=1:3], "bc")
    assert_equal(s[grapheme=2], "c")
    # as_bytes is a byte Span; codepoints and graphemes are iterators
    assert_equal(len(String("abc").as_bytes()), 3)
    var cps = 0
    for _ in String("héllo").codepoints():
        cps += 1
    assert_equal(cps, 5)
    var gs = 0
    for _ in String("héllo").graphemes():
        gs += 1
    assert_equal(gs, 5)


def test_parse_failures() raises:
    # Int(s) raises on a float, hex, empty, or out-of-range string
    with assert_raises():
        _ = Int("3.5")
    with assert_raises():
        _ = Int("0xff")
    with assert_raises():
        _ = Int("")
    with assert_raises():
        _ = Int("9223372036854775808")  # Int.MAX + 1
    # Float64(s) accepts exponents and inf, raises on empty or garbage
    assert_equal(Float64("1e3"), 1000.0)
    assert_equal(Float64("inf"), Float64.MAX * 2)  # overflows to +inf
    with assert_raises():
        _ = Float64("")
    with assert_raises():
        _ = Float64("garbage")


def test_utf8() raises:
    var bad: List[Byte] = [0x68, 0xFF, 0x69]  # "h", an invalid byte, "i"
    with assert_raises():
        _ = String(from_utf8=Span(bad))
    var lossy = String(from_utf8_lossy=Span(bad))  # bad byte -> U+FFFD
    assert_equal(lossy, "h\uFFFDi")
    assert_equal(String(from_utf8="héllo".as_bytes()), "héllo")


def test_pointers() raises:
    # reach into a container's raw buffer, then vectorize it (the escape hatch)
    var r = List(range(4))
    var vec = r.unsafe_ptr().unsafe_load[width=4]()  # 4 elements -> one SIMD
    assert_equal(vec.reduce_add(), 6)  # 0+1+2+3
    assert_equal(r.unsafe_ptr()[unsafe_offset=0], 0)  # deref one element
    assert_equal(r.unsafe_ptr().unsafe_offset(2)[], 2)  # arithmetic, then deref
    var none = OptionalPointer[Int, MutUntrackedOrigin]()  # nullable pointer
    assert_false(Bool(none))


def test_collection_literals() raises:
    var arr = [1, 2, 3]  # unannotated brackets make an Array
    assert_equal(
        String(reflect[type_of(arr)].name()),
        String(reflect[Array[Int, 3]].name()),
    )
    assert_equal(arr[0], 1)
    var lst: List = [1, 2, 3]  # annotate to get a List
    assert_equal(
        String(reflect[type_of(lst)].name()), String(reflect[List[Int]].name())
    )
    assert_equal(len(lst), 3)
    var d = {"a": 1, "b": 2}  # unannotated braces make a Dict
    assert_equal(
        String(reflect[type_of(d)].name()),
        String(reflect[Dict[String, Int]].name()),
    )
    assert_equal(d["b"], 2)


def test_array() raises:
    # make: an unannotated bracket literal is an Array
    var lit = [1, 2, 3]
    assert_equal(lit[2], 3)
    assert_equal(
        String(reflect[type_of(lit)].name()),
        String(reflect[Array[Int, 3]].name()),
    )
    assert_equal(Array[Int, 3](fill=7)[2], 7)
    var squares = Array[Int, 4](fill_with=lambda (i: Int) -> Int: i * i)
    assert_equal(squares[3], 9)
    assert_equal(Array[Int, 2]()[1], 0)  # default values
    var raw = Array[Int, 2](uninitialized=True)
    raw[0] = 5
    raw[1] = 6
    assert_equal(raw[0] + raw[1], 11)
    # convert
    var a: Array[Int32, 4] = [10, 20, 30, 40]
    assert_equal(len(List(a)), 4)  # growable copy
    assert_equal(len(Span(a)), 4)  # view
    var joined = [1, 2].concat([3, 4, 5])  # moves both in
    assert_equal(len(joined), 5)
    var b = a.copy()  # explicit copy, independent storage
    b[0] = 99
    assert_equal(a[0], 10)
    # access: inline, contiguous storage like a C array
    assert_equal(size_of[Array[Int32, 4]](), 4 * size_of[Int32]())
    var p = a.unsafe_ptr()
    assert_equal(p.unsafe_offset(3)[], a[3])
    assert_equal(Int(p.unsafe_offset(1)) - Int(p), size_of[Int32]())
    # a type-erased pointer reaches C as void *: memset(void*, int, size_t)
    var buf = Array[UInt8, 8](fill=7)
    _ = external_call["memset", OpaquePointer[MutAnyOrigin]](
        buf.unsafe_ptr().unsafe_bitcast[NoneType](), Int32(0), 8
    )
    assert_equal(buf[7], 0)


def test_array_list_moves() raises:
    var l: List[Int] = [7, 8, 9]
    var from_list = Array[Int, 3](
        fill_with=lambda (i: Int) {imm l} -> Int: l[i]
    )
    assert_equal(from_list[2], 9)  # List -> Array; you pick n
    assert_equal(len(Span(l)), 3)  # view a List, no copy
    assert_equal(len(Span(from_list)), 3)  # view an Array, no copy


def test_simd_array_list() raises:
    # List -> Array: view the buffer as an Array, then copy it out
    var src: List[Int] = [1, 2, 3, 4]
    var viewed = src.unsafe_ptr().unsafe_bitcast[Array[Int, 4]]()[].copy()
    assert_equal(viewed[3], 4)
    var names: List[String] = ["a", "b"]  # non-numeric Copyable T
    var named = names.unsafe_ptr().unsafe_bitcast[Array[String, 2]]()[].copy()
    assert_equal(named[1], "b")
    # N comes from the Array's type, known at compile time
    comptime N = type_of(viewed).length
    assert_equal(N, 4)
    assert_equal(src.unsafe_ptr().unsafe_load[width=N]()[2], 3)
    var v = SIMD[DType.float32, 4](1, 2, 3, 4)
    var a = Array[Float32, 4](uninitialized=True)
    a.unsafe_ptr().unsafe_store(v)  # SIMD -> Array: one vector store
    assert_equal(a[3], 4.0)
    var back = a.unsafe_ptr().unsafe_load[width=4]()  # Array -> SIMD
    assert_true(back == v)
    var l = List(a)
    var lv = l.unsafe_ptr().unsafe_load[width=4]()  # List -> SIMD
    assert_true(lv == v)
    l.unsafe_ptr().unsafe_store(lv * 10)  # SIMD -> List
    assert_equal(l[1], 20.0)
    var lanes = Array[Float32, 4](
        fill_with=lambda (i: Int) {imm v} -> Float32: v[i]
    )
    assert_equal(lanes[0], 1.0)  # lane by lane


def test_list() raises:
    var lst: List[Int] = [1, 2, 3]
    assert_equal(len(lst), 3)
    assert_equal(lst[0], 1)  # one element, by reference
    var view: Span[Int, _] = lst[0:2]  # a Span view, no copy
    assert_equal(len(view), 2)
    var filled = List[Int](length=4, fill=0)
    assert_equal(len(filled), 4)
    assert_equal(filled[2], 0)
    var r = List(range(5))  # range -> List
    assert_equal(len(r), 5)
    assert_equal(r[4], 4)
    var roomy = List[Int](capacity=8)  # empty, with room for 8
    assert_equal(len(roomy), 0)
    assert_true(roomy.capacity() >= 8)
    var copied = List(lst)  # from any iterable
    assert_equal(len(copied), 3)
    var alloc = lst.unsafe_take_allocation()  # the buffer moves out
    var left = len(lst)
    dealloc(alloc^)  # free before any assert can raise
    assert_equal(left, 0)


def test_dict() raises:
    var d = Dict[Int, String]()
    d[1] = "one"  # fill with d[k] = v
    d[2] = "two"
    assert_equal(d.setdefault(3, "three"), "three")  # insert-if-absent
    assert_true(d.get(1))  # Optional[V]
    assert_false(d.get(99))
    assert_true(1 in d)
    assert_equal(d.pop(2), "two")  # value, removes it
    with assert_raises():
        _ = d.pop(2)  # already removed
    assert_equal(len(List(d.values())), 2)  # materialize the iterator
    ref slot = d.setdefault(4, "four")  # a ref into the Dict
    slot = "FOUR"
    assert_equal(d[4], "FOUR")
    assert_equal(d.find(1).value(), "one")  # find is Optional[V] too
    assert_equal(d.get(1, "none"), "one")  # value, or default
    assert_equal(d.get(99, "none"), "none")
    var keyed = Dict[String, Int].fromkeys(["a", "b"], 0)  # every key -> 0
    assert_equal(len(keyed), 2)
    assert_equal(keyed["b"], 0)


def test_optional() raises:
    var o = Optional(5)  # from a value
    assert_true(Bool(o))
    assert_equal(o.value(), 5)  # ref
    assert_equal(o[], 5)  # ref via []
    var empty = Optional[Int](None)  # empty
    assert_false(Bool(empty))
    var empty2 = Optional[Int]()  # empty, explicit
    assert_false(Bool(empty2))
    assert_equal(empty.or_else(99), 99)  # value, or default
    var seven = Optional(7)
    assert_equal(seven.take(), 7)  # move value out
    with assert_raises():
        _ = empty[]  # [] raises on empty; value() and take() abort


def main() raises:
    test_establishing_types()
    test_casting()
    test_numbers()
    test_numeric_creation()
    test_numeric_conversion()
    test_simd()
    test_string()
    test_parse_failures()
    test_utf8()
    test_pointers()
    test_collection_literals()
    test_array()
    test_array_list_moves()
    test_simd_array_list()
    test_list()
    test_dict()
    test_optional()
