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
# test_function_declarations.mojo
# Tests for function-declarations.mdx code examples.
# Skip: inferred_type[Int](5) positional error, keyword type
#        mismatch error, div keyword error, configure positional
#        error, sum missing keyword error, bad default ordering
#        error, mut default error, out + return type error,
#        variadic default error, out variadic error, __init__
#        without out error, where on runtime arg error,
#        Resource/__deinit__ (undefined _release), parse/parse_strict
#        (signature-only stubs), copy/move constructor snippets
#        (no enclosing struct), configure keyword-only (pass-only
#        body), overload-set import-shadowing snippet (fictional
#        `from somewhere import add` - hand tested), ambiguous overload
#        call `foo("Hello")` (negative example, exercised by the
#        disambiguation test instead), return-type-only overload
#        of `parse` (negative example: `redefinition of function
#        'parse': cannot overload on return type only`),
#        raises-only overload of `maybe_raise` (negative example:
#        `redefinition of function 'maybe_raise' with identical
#        signature` - hand tested), literal-conversion ambiguity
#        of `take_param[1, 2]()` against `[Int, Int]` vs
#        `[Int, Float64]` (negative example; the runnable
#        `take_param` test uses `[Int, String]` instead),
#        unguarded `var self` on a `TrivialRegisterPassable` type
#        (sharp edge: `^` is a no-op for TRP, so `(c^).which()`
#        returns the `ref self` value - hand verified; the runnable
#        test instead uses `where not conforms_to(Self,
#        TrivialRegisterPassable)` to filter the overload out).
from std.testing import assert_equal
from std.reflection import reflect

# --- Function overloads: basic ---


def add_pair(x: Int, y: Int) -> Int:
    return x + y


def add_pair(x: String, y: String) -> String:
    return x + y


def test_add_pair() raises:
    assert_equal(add_pair(1, 2), 3)
    assert_equal(add_pair("Hi, ", "Mojo"), "Hi, Mojo")


# --- Overloading parameters ---


def take_param[a: Int, b: Int]() -> String:
    return "take_param[a: Int, b: Int]"


def take_param[a: Int, b: String]() -> String:
    return "take_param[a: Int, b: String]"


def test_take_param() raises:
    assert_equal(take_param[1, 2](), "take_param[a: Int, b: Int]")
    assert_equal(take_param[1, "hi"](), "take_param[a: Int, b: String]")


# --- Overloading the `self` convention ---


@fieldwise_init
struct Counter(Copyable):
    var n: Int

    def which(ref self) -> Int:
        return 1

    # Constrain `var self` to types where `^` actually transfers.
    # For `TrivialRegisterPassable` types, `^` is a no-op, so this
    # overload would otherwise be silently unreachable.
    def which(
        var self,
    ) -> Int where not conforms_to(Self, TrivialRegisterPassable):
        return 2


@fieldwise_init
struct CounterRegister(ImplicitlyCopyable, TrivialRegisterPassable):
    var n: Int

    def which(ref self) -> Int:
        return 1

    # Same constraint here: the `where` clause filters out this
    # overload entirely for `TrivialRegisterPassable` types.
    def which(
        var self,
    ) -> Int where not conforms_to(Self, TrivialRegisterPassable):
        return 2


def test_self_convention_overload() raises:
    # Non-TRP: both overloads reachable
    var c1 = Counter(0)
    assert_equal(c1.which(), 1)  # default call borrows -> ref self

    var c2 = Counter(0)
    assert_equal((c2^).which(), 2)  # transfer at call site -> var self

    # TRP: the `var self` overload is excluded by `where`, so only
    # `ref self` matches any call.
    var r1 = CounterRegister(0)
    assert_equal(r1.which(), 1)


# --- Instance methods beat static methods ---


struct StaticOverload:
    def __init__(out self):
        pass

    def foo(mut self) -> String:
        return "instance method"

    @staticmethod
    def foo() -> String:
        return "static method"


def test_instance_beats_static() raises:
    var a = StaticOverload()
    assert_equal(a.foo(), "instance method")
    assert_equal(StaticOverload.foo(), "static method")


# --- Variadic candidates lose ties ---


def take_one_or_many(x: Int) -> String:
    return "take(x: Int)"


def take_one_or_many(*xs: Int) -> String:
    return "take(*xs: Int)"


def test_variadic_ties() raises:
    assert_equal(take_one_or_many(1), "take(x: Int)")
    assert_equal(take_one_or_many(1, 2, 3), "take(*xs: Int)")


# --- Ambiguity, resolved by cast ---


struct MyString:
    @implicit
    def __init__(out self, s: String):
        pass


struct YourString:
    @implicit
    def __init__(out self, s: String):
        pass


def overloaded_str(name: MyString) -> String:
    return "MyString"


def overloaded_str(name: YourString) -> String:
    return "YourString"


def test_ambiguity_resolved_by_cast() raises:
    # overloaded_str("Hello") would be ambiguous: skipped intentionally.
    assert_equal(overloaded_str(MyString("Hello")), "MyString")
    assert_equal(overloaded_str(YourString("Hello")), "YourString")


def main() raises:
    test_add_pair()
    test_take_param()
    test_self_convention_overload()
    test_instance_beats_static()
    test_variadic_ties()
    test_ambiguity_resolved_by_cast()
