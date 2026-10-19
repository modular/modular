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
# test_compound_statements.mojo
# Tests for compound-statements.mdx code examples.
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true


# --- If statements ---


def test_if_elif_else_chain() raises:
    var result: String
    var x = 5
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    else:
        result = "you should never get here"
    assert_equal(result, "positive")

    x = -3
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    assert_equal(result, "negative")

    x = 0
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    assert_equal(result, "zero")


def test_single_line_if() raises:
    var result = String("placeholder")
    var x = 5
    if x > 0:
        result = "positive"
    assert_equal(result, "positive")


# --- Match statements ---


def _status_text(status_code: Int) -> String:
    match status_code:
        case 200:
            return "ok"
        case 301 | 302:
            return "redirect"
        case code if code >= 500:
            return String("server error ", code)
        case _:
            return "other"


def test_match_basic() raises:
    assert_equal(_status_text(200), "ok")
    assert_equal(_status_text(302), "redirect")
    assert_equal(_status_text(503), "server error 503")
    # A false guard continues to the next case.
    assert_equal(_status_text(418), "other")


def _next_code(mut calls: Int) -> Int:
    calls += 1
    return 7


def test_match_subject_evaluated_once() raises:
    var calls = 0
    var result: String
    match _next_code(calls):
        case 1:
            result = "one"
        case 2:
            result = "two"
        case _:
            result = "other"
    assert_equal(result, "other")
    assert_equal(calls, 1)


def _first_wins_hits(code: Int) -> Int:
    var hits = 0
    match code:
        case 2:
            hits += 1
        case value if value > 0:
            hits += 10
    return hits


def test_match_first_case_wins() raises:
    # 2 also satisfies the guard; only the first matching body runs.
    assert_equal(_first_wins_hits(2), 1)


def _single_case_hits(code: Int) -> Int:
    var hits = 0
    match code:
        case 1:
            hits += 1
    return hits


def test_match_no_case_runs_nothing() raises:
    assert_equal(_single_case_hits(9), 0)


# fmt: off
def _aligned_cases(code: Int) -> String:
    match code:
    case 2:
        return "two"
    case _:
        return "other"
# fmt: on


def test_match_case_alignment() raises:
    assert_equal(_aligned_cases(2), "two")


# --- Patterns ---


@fieldwise_init
struct Point(Copyable, Equatable, Writable):
    var x: Int
    var y: Int


def _compare_to(threshold: Int, sample: Int) -> String:
    match sample:
        case == threshold:
            return "at threshold"
        case _:
            return "off threshold"


def _quadrant(x: Int, y: Int) -> String:
    match x > 0, y > 0:
        case True, True:
            return "I"
        case False, True:
            return "II"
        case False, False:
            return "III"
        case True, False:
            return "IV"


def _corner(x: Int, y: Int) -> String:
    match (x, y):
        case (0, 0):
            return "origin"
        case _:
            return "other"


def _struct_kind(p: Point) -> String:
    match p:
        case Point(x=0, y=0):
            return "origin"
        case {x = 1, y = 1}:
            return "unit"
        case _:
            return "other"


def _describe_marker(marker: Optional[Point]) -> String:
    match marker:
        case .Some(Point(x=0, y=0)):
            return "at origin"
        case .Some(Point(x=x, y=y)) if x == y:
            return "on diagonal"
        case .None:
            return "no marker"
        case _:
            return "elsewhere"


def _summarize(samples: List[Int]) -> String:
    match samples:
        case []:
            return "empty"
        case [first, *middle, last]:
            return String(first, " ", len(middle), " ", last)
        case _:
            return "single"


def test_value_pattern() raises:
    assert_equal(_compare_to(10, 10), "at threshold")
    assert_equal(_compare_to(10, 3), "off threshold")


def test_tuple_patterns() raises:
    assert_equal(_quadrant(1, 1), "I")
    assert_equal(_quadrant(-1, 1), "II")
    assert_equal(_quadrant(-1, -1), "III")
    assert_equal(_quadrant(1, -1), "IV")
    assert_equal(_corner(0, 0), "origin")
    assert_equal(_corner(0, 1), "other")


def test_struct_and_initializer_list_patterns() raises:
    assert_equal(_struct_kind(Point(0, 0)), "origin")
    assert_equal(_struct_kind(Point(1, 1)), "unit")
    assert_equal(_struct_kind(Point(2, 0)), "other")


def test_nested_patterns() raises:
    assert_equal(_describe_marker(Optional(Point(0, 0))), "at origin")
    assert_equal(_describe_marker(Optional(Point(3, 3))), "on diagonal")
    assert_equal(_describe_marker(None), "no marker")
    assert_equal(_describe_marker(Optional(Point(1, 2))), "elsewhere")


def test_list_patterns() raises:
    assert_equal(_summarize([]), "empty")
    assert_equal(_summarize([1, 2, 3, 4]), "1 2 4")
    # A length-1 list fails [first, *middle, last], so it falls through.
    assert_equal(_summarize([5]), "single")


# --- Bindings ---


def _check_limit(reading: Int) -> Int:
    var limit = 100
    match reading:
        case == limit:
            return -1
        case limit:
            return limit


def test_bare_name_binds_new_name() raises:
    assert_equal(_check_limit(100), -1)
    # The bare `limit` binds the reading instead of comparing with 100.
    assert_equal(_check_limit(42), 42)


def test_ref_and_var_bindings() raises:
    var marker = Point(4, 0)
    match marker:
        case Point(x=ref x, y=0):
            x = 0
        case Point(x=var x, y=_):
            x += 1
    assert_equal(marker.x, 0)  # ref wrote through

    var other = Point(4, 2)
    var copied = 0
    match other:
        case Point(x=ref x, y=0):
            x = 0
        case Point(x=var x, y=_):
            x += 1
            copied = x
    assert_equal(copied, 5)
    assert_equal(other.x, 4)  # var changed only the copy


def _as_binding(status_code: Int) -> Int:
    match status_code:
        case 404 as code:
            return code
        case _:
            return 0


def test_as_binding() raises:
    assert_equal(_as_binding(404), 404)
    assert_equal(_as_binding(200), 0)


# --- Conditional patterns ---


def _pick(primary: Optional[Int], backup: Optional[Int]) -> String:
    if case .Some(reading) = primary:
        return String("primary ", reading)
    elif case .Some(reading) = backup:
        return String("backup ", reading)
    else:
        return "no reading"


def test_if_case() raises:
    assert_equal(_pick(Optional(1), Optional(2)), "primary 1")
    assert_equal(_pick(None, Optional(2)), "backup 2")
    assert_equal(_pick(None, None), "no reading")


struct JobQueue:
    var jobs: List[Int]

    def __init__(out self, var jobs: List[Int]):
        self.jobs = jobs^

    def next_job(mut self) -> Optional[Int]:
        if self.jobs:
            return self.jobs.pop()
        return None


def test_while_case() raises:
    var queue = JobQueue([3, 2, 1])
    var total = 0
    var else_runs = 0
    while case .Some(job) = queue.next_job():
        total += job
    else:
        else_runs += 1
    assert_equal(total, 6)
    assert_equal(else_runs, 1)


# --- While loops ---


def test_while_basic() raises:
    var count = 0
    var total = 0
    while count < 10:
        total += count
        count += 1
    assert_equal(count, 10)
    assert_equal(total, 45)  # 0+1+...+9


def test_while_break_continue() raises:
    # break exits the loop, continue skips to the next iteration.
    var processed = 0
    var i = 0
    while True:
        if i >= 10:
            break
        if i % 2 == 1:
            i += 1
            continue
        processed += 1
        i += 1
    assert_equal(processed, 5)  # 0, 2, 4, 6, 8


# --- For loops ---


def test_for_conventions() raises:
    var list: List[String] = ["a", "b", "c", "d"]

    for var item in list:
        item = item + "x"  # works. item is mutable copy of list element
        # print(item) # prints "ax", then "bx", "cx", and "dx"
        assert_true(item.endswith("x"))
    assert_true(list == ["a", "b", "c", "d"])  # list is unchanged
    # print(list) # unchanged

    for ref item in list:
        item = item + "x"
        # mutability picked up in reference to list element
        # print(item) # prints "ax", then "bx", "cx", and "dx"
        assert_true(item.endswith("x"))
    assert_true(list == ["ax", "bx", "cx", "dx"])  # list is changed
    # print(list) # changed to ["ax", "bx", "cx", "dx"]


def test_for_range() raises:
    var total = 0
    for i in range(10):
        total += i
    assert_equal(total, 45)


def test_for_destructuring() raises:
    # Tuple unpacking in the loop target.
    var pairs = [("a", 1), ("b", 2), ("c", 3)]
    var total = 0
    for _key, value in pairs:
        total += value
    assert_equal(total, 6)


# --- For/while else clauses ---


def test_for_else_no_break() raises:
    # else runs when the loop completes without break.
    var items = [1, 2, 3]
    var target = 99
    var found = False
    var ran_else = False
    for item in items:
        if item == target:
            found = True
            break
    else:
        ran_else = True
    assert_false(found)
    assert_true(ran_else)


def test_for_else_with_break() raises:
    # else does NOT run when break exits the loop.
    var items = [1, 2, 3]
    var target = 2
    var found = False
    var ran_else = False
    for item in items:
        if item == target:
            found = True
            break
    else:
        ran_else = True
    assert_true(found)
    assert_false(ran_else)


def test_while_else_no_break() raises:
    var i = 0
    var ran_else = False
    assert_false(ran_else)  # sentinel: must transition to True via else
    while i < 5:
        i += 1
    else:
        ran_else = True
    assert_equal(i, 5)
    assert_true(ran_else)


# --- Error handling ---


def _risky() raises:
    raise Error("oops")


def test_try_except_basic() raises:
    var caught = False
    try:
        _risky()
    except:
        caught = True
    assert_true(caught)


def test_try_full_structure_no_error() raises:
    # try -> else -> finally when no error fires.
    var sequence = String("")
    try:
        sequence += "T"
    except _:
        sequence += "E"
    else:
        sequence += "L"
    finally:
        sequence += "F"
    assert_equal(sequence, "TLF")


def test_try_full_structure_with_error() raises:
    # try -> except -> finally when an error fires.
    var sequence = String("")
    try:
        sequence += "T"
        _risky()
    except e:
        sequence += "E"
    else:
        sequence += "L"
    finally:
        sequence += "F"
    assert_equal(sequence, "TEF")


def test_except_with_binding() raises:
    var msg = String("")
    try:
        _risky()
    except e:
        msg = String(e)
    assert_true("oops" in msg)


def test_except_without_binding() raises:
    var caught = False
    try:
        _risky()
    except:
        caught = True
    assert_true(caught)


# --- Typed errors ---


@fieldwise_init
struct NetworkError(ImplicitlyCopyable, Movable, Writable):
    var message: String
    var code: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("NetworkError(code=", self.code, "): ", self.message)


def _fetch() raises NetworkError -> String:
    raise NetworkError("HTCPCP", 418)


def test_typed_raises() raises:
    # raises Type infers the bound error's type and allows field access.
    var code = 0
    var msg = String("placeholder")
    try:
        _ = _fetch()
    except e:
        code = e.code
        msg = e.message
    assert_equal(code, 418)
    assert_equal(msg, "HTCPCP")


# --- Context managers ---


struct Scope(ImplicitlyCopyable):
    var label: String

    def __init__(out self, label: String):
        self.label = label

    def __enter__(self) -> Self:
        # perform setup tasks
        # print("entering", self.label) Not needed for testing
        return self

    def __exit__(self):
        # perform cleanup tasks
        # print("exiting", self.label) Not needed for testing
        pass


def test_custom_context_manager() raises:
    with Scope("setup") as s:
        assert_equal(s.label, "setup")


# --- comptime if ---


def test_comptime_if() raises:
    # Either branch may be selected depending on the target.
    var label: String
    comptime if size_of[Int]() == 8:
        label = "64-bit"
    else:
        label = "Probably 32-bit"
    assert_true(label == "64-bit" or label == "Probably 32-bit")


# --- comptime for ---


def test_comptime_for() raises:
    # The loop unrolls at compile time; body emits as total += 0; total += 1; ...
    var total = 0
    comptime for i in range(5):
        total += i
    assert_equal(total, 10)  # 0+1+2+3+4


# --- comptime match ---


def _element_kind[T: AnyType]() -> String:
    comptime match T:
        case == Int:
            return "integer"
        case == String:
            return "string"
        case _:
            return "other"


def test_comptime_match() raises:
    assert_equal(_element_kind[Int](), "integer")
    assert_equal(_element_kind[String](), "string")
    assert_equal(_element_kind[Float64](), "other")


# --- Closures with capture lists ---


def test_closure_with_capture_list() raises:
    # {mut count} captures by mutable reference.
    var count = 0

    def inner() {mut count}:
        count += 1

    inner()
    inner()
    inner()
    assert_equal(count, 3)


def main() raises:
    test_if_elif_else_chain()
    test_single_line_if()
    test_match_basic()
    test_match_subject_evaluated_once()
    test_match_first_case_wins()
    test_match_no_case_runs_nothing()
    test_match_case_alignment()
    test_value_pattern()
    test_tuple_patterns()
    test_struct_and_initializer_list_patterns()
    test_nested_patterns()
    test_list_patterns()
    test_bare_name_binds_new_name()
    test_ref_and_var_bindings()
    test_as_binding()
    test_if_case()
    test_while_case()
    test_while_basic()
    test_while_break_continue()
    test_for_conventions()
    test_for_range()
    test_for_destructuring()
    test_for_else_no_break()
    test_for_else_with_break()
    test_while_else_no_break()
    test_try_except_basic()
    test_try_full_structure_no_error()
    test_try_full_structure_with_error()
    test_except_with_binding()
    test_except_without_binding()
    test_typed_raises()
    test_custom_context_manager()
    test_comptime_if()
    test_comptime_for()
    test_comptime_match()
    test_closure_with_capture_list()
