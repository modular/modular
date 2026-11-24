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
# tests.mojo
# Tests for functions.mdx code examples.
#
# Not tested (intentional): the page's 13 `no-test` blocks are negative or
# illustrative examples -- undefined placeholder types, two spellings of one
# signature shown as equivalent, bodyless sketches, unsupported syntax, and
# deliberate compile errors. A runtime test can't assert a compile error.
#
# Reconciled to coexist in one module:
#   - `my_pow()` is defined once, under `optional-args`. The page repeats the
#     declaration under "Keyword arguments" so that example reads on its own;
#     that block is tagged `duplicate-of=optional-args` in the MDX.
#   - The page's two `make_worldly()` bodies are named `make_worldly()` and
#     `make_worldly2()`, since one module can't define the same signature
#     twice.
#   - "Anatomy of a function" uses `max()`; `add()` belongs to the overload
#     set under "Overloaded functions".
#
# `main()` is deliberately not `raises`: the `count-many-things` doc block
# shows a plain `def main():`. Assertions run through `run_tests()`, which
# catches and exits nonzero so a failure still fails the binary.
from std.collections import StringDict
from std.sys import exit
from std.testing import assert_equal, assert_raises


# --- Anatomy of a function ---


# start-do-nothing
def do_nothing():
    pass
    # end-do-nothing


# start-max-args
def max(a: Int, b: Int) -> Int:
    return a if a > b else b
    # end-max-args


def test_positional_arguments() raises:
    assert_equal(max(5, 7), 7)


def test_keyword_arguments_in_any_order() raises:
    assert_equal(max(b=3, a=9), 9)
    assert_equal(max(a=9, b=3), 9)


def test_positional_and_keyword_arguments_mixed() raises:
    assert_equal(max(5, b=7), 7)


def test_pass_body_is_a_complete_definition() raises:
    do_nothing()


def greet_implicit(name: String):
    pass


def greet_none(name: String) -> None:
    pass


def test_omitted_and_explicit_none_return_types_are_equivalent() raises:
    greet_implicit("Sam")
    greet_none("Sam")


# --- Optional arguments ---


# start-optional-args
def my_pow(base: Int, exp: Int = 2) -> Int:
    return base**exp


def use_defaults():
    # Uses the default value for `exp`
    var z = my_pow(3)
    print(z)
    # end-optional-args


def test_omitted_argument_uses_its_default() raises:
    assert_equal(my_pow(3), 9)


def test_supplied_argument_overrides_the_default() raises:
    assert_equal(my_pow(3, 3), 27)


# --- Keyword arguments ---


# start-keyword-args
def use_keywords():
    # Uses keyword argument names (with order reversed)
    var z = my_pow(exp=3, base=2)
    print(z)
    # end-keyword-args


def test_keyword_arguments_may_be_reordered() raises:
    assert_equal(my_pow(exp=3, base=2), 8)
    assert_equal(my_pow(base=2, exp=3), 8)


def test_keyword_form_matches_the_positional_form() raises:
    assert_equal(my_pow(exp=3, base=2), my_pow(2, 3))


# --- Homogeneous variadic arguments ---


# start-variadic-greet
def greet(*names: String):
    ...
    # end-variadic-greet


# start-variadic-sum
def sum(*values: Int) -> Int:
    var sum: Int = 0
    for value in values:
        sum = sum + value
    return sum
    # end-variadic-sum


def test_sum_of_several_values() raises:
    assert_equal(sum(1, 2, 3), 6)


def test_variadic_accepts_any_number_of_arguments() raises:
    assert_equal(sum(), 0)
    assert_equal(sum(7), 7)
    assert_equal(sum(1, 2, 3, 4, 5), 15)


def test_greet_accepts_any_number_of_names() raises:
    greet()
    greet("Alice")
    greet("Alice", "Bob")


def list_len(values: VariadicList[Int, _]) -> Int:
    return len(values)


def variadic_list_len(*values: Int) -> Int:
    return list_len(values)


def after_label(label: String, *values: Int) -> Int:
    return list_len(values)


def test_variadic_argument_is_a_variadic_list() raises:
    # Binding the argument to a `VariadicList` parameter stops compiling if
    # the manual's claim about the in-function type ever goes stale.
    assert_equal(variadic_list_len(1, 2, 3), 3)


def test_arguments_may_precede_the_variadic_argument() raises:
    assert_equal(after_label("count", 1, 2), 2)
    assert_equal(after_label("count"), 0)


# --- Mutating a variadic argument ---


# start-variadic-mut-ref
def make_worldly(mut *strs: String):
    for ref i in strs:
        i += " world"
        # end-variadic-mut-ref


# start-variadic-mut-index
def make_worldly2(mut *strs: String):
    for i in range(len(strs)):
        strs[i] += " world"
        # end-variadic-mut-index


def test_ref_binding_mutates_the_callers_values() raises:
    var a = String("hello")
    var b = String("hi")
    make_worldly(a, b)
    assert_equal(a, "hello world")
    assert_equal(b, "hi world")


def test_indexing_mutates_the_callers_values() raises:
    var a = String("hello")
    var b = String("hi")
    make_worldly2(a, b)
    assert_equal(a, "hello world")
    assert_equal(b, "hi world")


# --- Variadic keyword arguments ---


# start-print-nicely
def print_nicely(var **kwargs: Int):
    for item in kwargs.items():
        print(item.key, "=", item.value)
        # end-print-nicely


def int_kwargs(var **kwargs: Int) -> StringDict[Int]:
    return kwargs^


def float_kwargs(var **kwargs: Float64) -> StringDict[Float64]:
    return kwargs^


def test_argument_dictionary_holds_the_passed_keywords() raises:
    var kwargs = int_kwargs(a=7, y=8)
    assert_equal(len(kwargs), 2)
    assert_equal(kwargs["a"], 7)
    assert_equal(kwargs["y"], 8)


def test_argument_dictionary_type_follows_the_argument_type() raises:
    # Both return annotations stop compiling if the dictionary is no longer a
    # `StringDict` of the declared argument type.
    assert_equal(float_kwargs(x=1.5)["x"], 1.5)


def test_any_number_of_keyword_arguments() raises:
    print_nicely()
    print_nicely(a=7, y=8)


# --- Positional-only and keyword-only arguments ---


# start-positional-only
def min(a: Int, b: Int, /) -> Int:
    return a if a < b else b
    # end-positional-only


# start-keyword-only-after-variadic
def sort(*values: Float64, ascending: Bool = True):
    ...
    # end-keyword-only-after-variadic


# start-keyword-only
def kw_only_args(a1: Int, a2: Int, *, double: Bool) -> Int:
    var product = a1 * a2
    if double:
        return product * 2
    else:
        return product
    # end-keyword-only


def test_positional_only_arguments() raises:
    assert_equal(min(1, 2), 1)
    assert_equal(min(2, 1), 1)
    assert_equal(min(3, 3), 3)


def test_required_keyword_only_argument() raises:
    assert_equal(kw_only_args(2, 3, double=True), 12)
    assert_equal(kw_only_args(2, 3, double=False), 6)


def test_keyword_only_argument_after_a_variadic_is_optional() raises:
    sort(1.1, 6.5, 4.3)
    sort(1.1, 6.5, 4.3, ascending=False)
    sort(ascending=True)


# --- Overloaded functions ---


# start-overloaded-add
def add(x: Int, y: Int) -> Int:
    return x + y


def add(x: String, y: String) -> String:
    return x + y
    # end-overloaded-add


def test_int_overload() raises:
    assert_equal(add(2, 3), 5)


def test_string_overload() raises:
    assert_equal(add(String("Hello, "), String("world")), "Hello, world")


def test_string_literal_reaches_the_string_overload() raises:
    # A `StringLiteral` converts implicitly to `String`, which also has to
    # select the `String` overload over the `Int` one.
    assert_equal(add("Hello, ", "world"), "Hello, world")


# --- Overload resolution: ambiguity resolved by an explicit cast ---


# start-overload-ambiguity
struct MyString:
    @implicit
    def __init__(out self, string: String):
        pass


struct YourString:
    @implicit
    def __init__(out self, string: String):
        pass


def foo(name: MyString):
    print("MyString")


def foo(name: YourString):
    print("YourString")


def call_foo():
    # Both `foo` overloads can accept `"Hello"`, so Mojo doesn't know
    # which one to call.
    foo(MyString("Hello"))
    # end-overload-ambiguity


def which(name: MyString) -> String:
    return "MyString"


def which(name: YourString) -> String:
    return "YourString"


def test_explicit_cast_selects_the_intended_overload() raises:
    # The manual's `foo()` overloads print instead of returning, so these
    # mirror them over the same two types to make the choice observable.
    assert_equal(which(MyString("Hello")), "MyString")
    assert_equal(which(YourString("Hello")), "YourString")


# --- Overload resolution: the precedence rules ---


struct Wrapper:
    @implicit
    def __init__(out self, value: Int):
        pass


def fewest_conversions(x: Int) -> String:
    return "exact"


def fewest_conversions(x: Wrapper) -> String:
    return "converted"


def test_fewest_implicit_conversions_wins() raises:
    assert_equal(fewest_conversions(1), "exact")


def no_variadic(x: Int) -> String:
    return "non-variadic"


def no_variadic(*x: Int) -> String:
    return "variadic"


def test_candidate_without_variadic_arguments_wins() raises:
    assert_equal(no_variadic(1), "non-variadic")


def shortest_signature(x: Int) -> String:
    return "no-parameters"


def shortest_signature[T: Intable](x: T) -> String:
    return "one-parameter"


def test_shortest_parameter_signature_wins() raises:
    assert_equal(shortest_signature(1), "no-parameters")


@fieldwise_init
struct OverloadedStruct:
    def foo(self) -> String:
        return "instance"

    @staticmethod
    def foo() -> String:
        return "static"


def test_non_staticmethod_candidate_wins() raises:
    var instance = OverloadedStruct()
    assert_equal(instance.foo(), "instance")
    # Calling on an rvalue, `OverloadedStruct().foo()`, currently selects the
    # static overload instead. Known compiler bug, deliberately not asserted.


# --- Return values ---


# start-get-greeting
def get_greeting() -> String:
    return "Hello"
    # end-get-greeting


def test_string_literal_converts_to_the_declared_return_type() raises:
    assert_equal(get_greeting(), "Hello")


# --- Named results ---


@fieldwise_init
struct NameTag:
    var name: String


# start-named-result
def get_name_tag(var name: String, out name_tag: NameTag):
    name_tag = NameTag(name^)
    # end-named-result


def tag_out_first(out tag: NameTag, var name: String):
    tag = NameTag(name^)


def tag_bare_return(var name: String, out tag: NameTag):
    tag = NameTag(name^)
    return


def tag_returns_a_value(var name: String, out tag: NameTag):
    tag = NameTag(String("ignored"))
    return NameTag(name^)


def test_named_result_reaches_the_caller() raises:
    assert_equal(get_name_tag("Judith").name, "Judith")


def test_out_argument_may_appear_first() raises:
    assert_equal(tag_out_first(name="Judith").name, "Judith")


def test_bare_return_yields_the_out_argument() raises:
    assert_equal(tag_bare_return("Judith").name, "Judith")


def test_returned_value_wins_over_the_out_argument() raises:
    # The `out` argument is initialized to something else first, so this only
    # passes if the explicit `return` value is what reaches the caller.
    assert_equal(tag_returns_a_value("Judith").name, "Judith")


# --- Named results: returning a type that can't be moved or copied ---


# start-immovable-object
struct ImmovableObject:
    var name: String

    def __init__(out self, var name: String):
        self.name = name^


def create_immovable_object(var name: String, out obj: ImmovableObject):
    obj = ImmovableObject(name^)
    obj.name += "!"
    # obj is implicitly returned
    # end-immovable-object


# start-immovable-direct-return
def create_immovable_object3(var name: String) -> ImmovableObject:
    return ImmovableObject(name^)  # OK
    # end-immovable-direct-return


def test_named_result_is_built_in_place() raises:
    assert_equal(create_immovable_object("Blob").name, "Blob!")


def test_value_returned_immediately_needs_no_move() raises:
    assert_equal(create_immovable_object3("Bob").name, "Bob")


# --- Raising and non-raising functions ---


# start-raises-error
def raises_error() raises:
    raise Error("There was an error.")
    # end-raises-error


# The page shows these alongside a third function that deliberately fails to
# compile, so the block is marked `no-test` and only these two are here.
def handle_error():
    try:
        raises_error()
    except e:
        print("Handled an error:", e)


def propagate_error() raises:
    raises_error()


def test_raises_error_propagates_its_message() raises:
    with assert_raises(contains="There was an error."):
        raises_error()


def test_handle_error_swallows_the_error() raises:
    handle_error()


def test_propagate_error_re_raises_to_its_caller() raises:
    with assert_raises(contains="There was an error."):
        propagate_error()


# --- Test runner ---


def run_tests():
    try:
        test_positional_arguments()
        test_keyword_arguments_in_any_order()
        test_positional_and_keyword_arguments_mixed()
        test_pass_body_is_a_complete_definition()
        test_omitted_and_explicit_none_return_types_are_equivalent()
        test_omitted_argument_uses_its_default()
        test_supplied_argument_overrides_the_default()
        test_keyword_arguments_may_be_reordered()
        test_keyword_form_matches_the_positional_form()
        test_sum_of_several_values()
        test_variadic_accepts_any_number_of_arguments()
        test_greet_accepts_any_number_of_names()
        test_variadic_argument_is_a_variadic_list()
        test_arguments_may_precede_the_variadic_argument()
        test_ref_binding_mutates_the_callers_values()
        test_indexing_mutates_the_callers_values()
        test_argument_dictionary_holds_the_passed_keywords()
        test_argument_dictionary_type_follows_the_argument_type()
        test_any_number_of_keyword_arguments()
        test_positional_only_arguments()
        test_required_keyword_only_argument()
        test_keyword_only_argument_after_a_variadic_is_optional()
        test_int_overload()
        test_string_overload()
        test_string_literal_reaches_the_string_overload()
        test_explicit_cast_selects_the_intended_overload()
        test_fewest_implicit_conversions_wins()
        test_candidate_without_variadic_arguments_wins()
        test_shortest_parameter_signature_wins()
        test_non_staticmethod_candidate_wins()
        test_string_literal_converts_to_the_declared_return_type()
        test_named_result_reaches_the_caller()
        test_out_argument_may_appear_first()
        test_bare_return_yields_the_out_argument()
        test_returned_value_wins_over_the_out_argument()
        test_named_result_is_built_in_place()
        test_value_returned_immediately_needs_no_move()
        test_raises_error_propagates_its_message()
        test_handle_error_swallows_the_error()
        test_propagate_error_re_raises_to_its_caller()
        test_documented_result()
        test_int_conversion_truncates()
        test_pack_accepts_any_number_of_arguments()
    except e:
        print("test failure:", e)
        exit(1)


# --- Heterogeneous variadic arguments ---
#
# `count_many_things()` must stay the last definition in this file. Its doc
# region runs from the `def` through `main()`'s first statement, so nothing
# may come between them.


def test_documented_result() raises:
    # The manual annotates this call with `# 28`.
    assert_equal(count_many_things(5, 11.7, 12), 28)


def test_int_conversion_truncates() raises:
    assert_equal(count_many_things(11.7), 11)


def test_pack_accepts_any_number_of_arguments() raises:
    assert_equal(count_many_things(), 0)
    assert_equal(count_many_things(1), 1)


# start-count-many-things
def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
    var total = 0

    comptime for i in range(args.__len__()):
        total += Int(args[i])

    return total


def main():
    print(count_many_things(5, 11.7, 12))  # 28
    # end-count-many-things

    # start-arg-passing
    # positional
    var x = max(5, 7)  # Positionally, a=5 and b=7
    # keyword
    var y = max(b=3, a=9)
    # mixed
    var z = max(5, b=7)  # Positionally, a=5
    # end-arg-passing
    print(x, y, z)

    # start-print-nicely-call
    # prints:
    # `a = 7`
    # `y = 8`
    print_nicely(a=7, y=8)
    # end-print-nicely-call

    # start-sort-call
    sort(1.1, 6.5, 4.3, ascending=False)
    # end-sort-call

    # start-named-result-call
    var tag = get_name_tag("Judith")
    # end-named-result-call
    print(tag.name)

    # start-immovable-object-call
    var my_obj = create_immovable_object("Blob")
    # end-immovable-object-call
    print(my_obj.name)

    do_nothing()
    use_defaults()
    use_keywords()
    greet("Alice", "Bob")
    print(sum(1, 2, 3))
    print(add(2, 3))
    print(get_greeting())
    print(create_immovable_object3("Bob").name)
    call_foo()
    handle_error()

    run_tests()
