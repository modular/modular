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
# test_compile_time.mojo
# Tests for the Mojo "compile-time" cheat-sheet card.
#
# Exercises the card's claims so one that drifts stops compiling or fails an
# assert: parameters, where clauses (with messages and on thin function
# types), trait bounds, running a function at compile time, literal
# precision, comptime if/for, sys.info queries with a
# CompilationTarget, comptime members, and the inlining decorators, including
# inlining chosen by a parameter.
#
# Not tested (no portable assertable value, or a compile error can't be
# asserted at runtime):
#   - reflect[T] field surface beyond .name(): newly introduced and documented
#     as incomplete; the wider field API may still shift.
#   - 2**200: overflows a 64-bit Int, so it can't be materialized to compare.
#   - comptime if on hardware facts (is_nvidia_gpu, ...): machine-dependent.
#   - CompilationTarget.default_accelerator(): fails to instantiate on a host
#     with no accelerator configured.
#   - where messages: they appear only in compile errors.
#   - the compile-time boundary (no file I/O, no raising, runs on CPU): these
#     are compile errors, which a runtime test can't assert.
from std.testing import assert_equal
from std.sys.info import size_of, align_of, simd_width_of
from std.sys.info import CompilationTarget
from std.builtin.globals import global_constant


# --- parameters: a value parameter and a type parameter ---
def scaled[factor: Int](x: Int) -> Int:
    return x * factor  # factor is a compile-time value parameter


def pick[T: Copyable](a: T, b: T, take_first: Bool) -> T:
    return a.copy() if take_first else b.copy()  # T is a type parameter


def test_parameters() raises:
    assert_equal(scaled[3](10), 30)
    assert_equal(pick[Int](1, 2, True), 1)


# --- where: gate on a numeric truth ---
def block[
    w: Int
]() -> Int where w.is_power_of_two() else ("'w' must be a power of two"):
    return w  # only callable when w is a power of two


# A thin function type carries the constraint; every bound function must too.
comptime Kernel = def[w: Int](Int) thin -> Int where w > 0 else (
    "'w' must be positive"
)


def times[w: Int](x: Int) -> Int where w > 0 else "'w' must be positive":
    return x * w


def apply[F: Kernel](x: Int) -> Int:
    return F[4](x)


# The type exists for every DType; average() exists only for numeric ones.
@fieldwise_init
struct Values[dtype: DType]:
    var a: Scalar[Self.dtype]
    var b: Scalar[Self.dtype]

    def average(self) -> Float64 where Self.dtype.is_numeric():
        return (Float64(self.a) + Float64(self.b)) / 2


def test_where() raises:
    assert_equal(block[8](), 8)
    assert_equal(apply[times](3), 12)
    assert_equal(Values[DType.int32](3, 4).average(), 3.5)
    _ = Values[DType.bool](True, False)  # constructs, but has no average()


# --- conformances: a trait bound admits only proven operations ---
def largest[T: Comparable & Copyable & Deinitable](xs: List[T]) -> T:
    var best_i = 0
    for i in range(len(xs)):
        if xs[i] > xs[best_i]:
            best_i = i
    return xs[best_i].copy()


def test_conformances() raises:
    assert_equal(largest([3, 1, 4, 1, 5]), 5)


# --- run code at compile time: any function, no marker ---
def square(x: Int) -> Int:
    return x * x


def test_run_at_compile_time() raises:
    comptime nine = square(3)  # square runs while compiling
    assert_equal(nine, 9)


# --- numeric precision: literals stay exact, Float64 rounds ---
def test_numeric_precision() raises:
    comptime c = 0.1 + 0.2  # folded exactly as a literal
    assert_equal(c == 0.3, True)
    var a = 0.1  # a Float64 now
    var r = a + 0.2
    assert_equal(r == 0.3, False)  # 0.30000000000000004


# --- query the target ---
def test_query_target() raises:
    assert_equal(size_of[Int32](), 4)
    assert_equal(size_of[Int64](), 8)
    assert_equal(align_of[Int32](), 4)
    comptime host = CompilationTarget.current()  # an explicit target
    assert_equal(size_of[Int64, host](), 8)
    assert_equal(
        simd_width_of[DType.float32, host](), simd_width_of[DType.float32]()
    )


# --- comptime members live on types ---
struct Stack[T: Copyable]:
    comptime Element = Self.T  # associated type (reached through Self)
    comptime capacity = 1024  # a compile-time value member


def test_comptime_members() raises:
    assert_equal(Stack[Int].capacity, 1024)
    var e: Stack[Int].Element = 5  # Element resolves to Int for Stack[Int]
    assert_equal(e, 5)


# --- comptime for: fully unrolled, index is a constant ---
def test_comptime_for() raises:
    var total = 0
    comptime for i in range(4):
        total += i  # 0 + 1 + 2 + 3
    assert_equal(total, 6)


# --- comptime if: only the live branch is kept ---
def test_comptime_if() raises:
    var width: Int
    comptime if size_of[Int64]() == 8:
        width = 64  # the live branch (condition is known)
    else:
        width = 0
    assert_equal(width, 64)


# --- inlining: force or forbid ---
@inline(.always)
def add_inline(a: Int, b: Int) -> Int:
    return a + b


@inline(.never)
def add_separate(a: Int, b: Int) -> Int:
    return a + b


@inline(.automatic)
def add_auto(a: Int, b: Int) -> Int:
    return a + b


@inline(.nodebug)
def add_nodebug(a: Int, b: Int) -> Int:
    return a + b


# --- inline by parameter: each instantiation chooses ---
@inline(policy)
def doubled[policy: InlineLevel](x: Int) -> Int:
    return x * 2


def test_inlining() raises:
    assert_equal(add_inline(2, 3), 5)
    assert_equal(add_separate(2, 3), 5)
    assert_equal(add_auto(2, 3), 5)
    assert_equal(add_nodebug(2, 3), 5)
    assert_equal(doubled[.always](3), 6)
    assert_equal(doubled[.never](3), 6)


# --- comptime __match: specialize by pattern ---
def small_tile[mt: Int, nt: Int]() -> String where mt * nt <= 256:
    return String("small ", mt, "x", nt)


def tile_path[m: Int, n: Int]() -> String:
    comptime __match (m, n):
        case (16, 16):
            return "tuned 16x16"
        case (mt, nt) if mt * nt <= 256:  # the guard proves small_tile's where
            return small_tile[mt, nt]()
        case _:
            return "generic"


def tile_count[m: Int]() -> Int:
    var hits = 0
    comptime __match m:
        case 16:
            hits += 1
    # no case _: an unmatched subject compiles to nothing
    return hits


def test_comptime_match() raises:
    assert_equal(tile_path[16, 16](), "tuned 16x16")
    assert_equal(tile_path[8, 4](), "small 8x4")
    assert_equal(tile_path[64, 64](), "generic")
    assert_equal(tile_count[16](), 1)
    assert_equal(tile_count[64](), 0)


# --- conditional availability: a method exists only when its condition holds ---
@fieldwise_init
struct Buf[n: Int]:
    var value: Int

    def first(self) -> Int where Self.n > 0:  # exists only when n > 0
        return self.value


def test_conditional_availability() raises:
    var b = Buf[3](42)
    assert_equal(b.first(), 42)


# --- type_of: capture the type of an expression ---
def test_type_of() raises:
    var x = 5
    var y: type_of(x) = x + 5  # type_of(x) is Int
    assert_equal(y, 10)


# --- conditional construction: default-construct only when proven ---
def test_conditional_construction() raises:
    var x = 0
    comptime if conforms_to(type_of(x), Defaultable):
        var y = type_of(x)()  # the default constructor (no make_default)
        assert_equal(y, x)  # both 0


# --- reflect: read a type's name ---
def test_reflect() raises:
    comptime name = reflect[Int].name()
    # Int unifies with SIMD, so its reflected name is the SIMD spelling.
    assert_equal(String(name), "SIMD[DType.int, 1]")


# --- materialization: comptime value -> runtime ---
comptime POWERS: Array[Int, 4] = [1, 2, 4, 8]


def test_materialization() raises:
    comptime table: List[Int] = [3, 5, 7, 11, 13]
    var t = materialize[table]()  # heap-backed comptime List -> runtime List
    assert_equal(t[1], 5)
    ref g = global_constant[POWERS]()  # static table, indexed without a copy
    assert_equal(g[2], 4)


def main() raises:
    test_parameters()
    test_where()
    test_conformances()
    test_run_at_compile_time()
    test_numeric_precision()
    test_query_target()
    test_comptime_members()
    test_comptime_for()
    test_comptime_if()
    test_inlining()
    test_conditional_availability()
    test_type_of()
    test_conditional_construction()
    test_reflect()
    test_materialization()
