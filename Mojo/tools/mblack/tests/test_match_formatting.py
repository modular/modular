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

from tests.util import assert_format, assert_mojo_format

import mblack


def test_mojo_keeps_case_indent():
    source = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    assert_mojo_format(source, source)


def test_mojo_indents_flat_cases():
    source = """\
def f(x):
    match x:
    case 1:
        return "one"
    case _:
        return "other"
"""
    expected = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_dunder_match():
    source = """\
def f(x):
    __match x:
        case 1:
            return "one"
        case 2 if x > 0:
            return "two"
        case _:
            return "other"
"""
    expected = """\
def f(x):
    __match x:
        case 1:
            return "one"
        case 2 if x > 0:
            return "two"
        case _:
            return "other"
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_explicit_value_patterns():
    source = """\
def f(value, expected):
    match value:
    case ==expected:
        return True
"""
    expected = """\
def f(value, expected):
    match value:
        case == expected:
            return True
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_if_case():
    source = """\
def f(pair):
    if case (0,x as same)=pair:
        return x
    elif case (x,1)=pair:
        return x
"""
    expected = """\
def f(pair):
    if case (0, x as same) = pair:
        return x
    elif case (x, 1) = pair:
        return x
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_multiple_if_clauses():
    source = """\
def f(enabled, user):
    if enabled;case .Some(value)=user;value.can_access():
        serve(value)
    elif enabled;case .Some(value)=fallback:
        serve(value)
"""
    expected = """\
def f(enabled, user):
    if enabled; case .Some(value) = user; value.can_access():
        serve(value)
    elif enabled; case .Some(value) = fallback:
        serve(value)
"""
    assert_mojo_format(source, expected)


def test_mojo_joins_multiple_if_clauses_that_fit():
    source = """\
def f(lhs, rhs):
    if case .Some(value) = lhs;
      value == rhs:
        return
"""
    expected = """\
def f(lhs, rhs):
    if case .Some(value) = lhs; value == rhs:
        return
"""
    assert_mojo_format(source, expected)


def test_mojo_splits_long_multiple_if_clauses():
    source = """\
def f(user):
    if is_feature_enabled(); case .Some(request) = next_request(); request.is_valid_for(user):
        process(request)
    else:
        dont_process()
"""
    expected = """\
def f(user):
    if is_feature_enabled();
      case .Some(request) = next_request();
      request.is_valid_for(user):
        process(request)
    else:
        dont_process()
"""
    assert_mojo_format(source, expected)


def test_mojo_keeps_continuation_indent_in_split_clause():
    source = """\
def f(a):
    if a; case .Some(value) = compute_something_long(argument_one, argument_two, argument_three):
        pass
"""
    expected = """\
def f(a):
    if a;
      case .Some(value) = compute_something_long(
          argument_one, argument_two, argument_three
      ):
        pass
"""
    assert_mojo_format(source, expected)


def test_mojo_splits_long_first_if_clause():
    # The single clause after the semicolon is a lone name, which the parser
    # does not wrap in a clause-list node.
    source = """\
def f():
    if a_very_long_condition_name_number_one(argument_one, argument_two, argument_three); b:
        pass
    x = 1
"""
    expected = """\
def f():
    if a_very_long_condition_name_number_one(
        argument_one, argument_two, argument_three
    );
      b:
        pass
    x = 1
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_multiple_guard_clauses():
    source = """\
def f(enabled, user):
    __guard enabled;case .Some(value)=user else:
        return
    __guard enabled;
      ready else:
        return
    serve(value)
"""
    expected = """\
def f(enabled, user):
    __guard enabled; case .Some(value) = user else:
        return
    __guard enabled; ready else:
        return
    serve(value)
"""
    assert_mojo_format(source, expected)


def test_mojo_splits_long_multiple_guard_clauses():
    source = """\
def f():
    __guard is_feature_enabled(); case .Some(request) = next_request(); request.ok() else:
        return
"""
    expected = """\
def f():
    __guard is_feature_enabled();
      case .Some(request) = next_request();
      request.ok() else:
        return
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_multiple_comptime_if_clauses():
    source = """\
def f[T: AnyType, enabled: Bool]():
    comptime if enabled;case ==Int=T:
        pass
    comptime if enabled_with_a_long_name; case == Int = T; long_condition_name[T]():
        pass
"""
    expected = """\
def f[T: AnyType, enabled: Bool]():
    comptime if enabled; case == Int = T:
        pass
    comptime if enabled_with_a_long_name;
      case == Int = T;
      long_condition_name[T]():
        pass
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_while_case():
    source = """\
def f(pair):
    while case ((0,x)|(x,1)) as matched=pair:
        pair = next_pair()
"""
    expected = """\
def f(pair):
    while case ((0, x) | (x, 1)) as matched = pair:
        pair = next_pair()
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_guard_case():
    source = """\
def f(opt):
    __guard case .Some(var value)=opt else:
        return
    use(value)
"""
    expected = """\
def f(opt):
    __guard case .Some(var value) = opt else:
        return
    use(value)
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_comptime_guard_case():
    source = """\
def f[T: AnyType]():
    comptime __guard case ==Int=T else:
        return
    use(T)
"""
    expected = """\
def f[T: AnyType]():
    comptime __guard case == Int = T else:
        return
    use(T)
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_guard_condition():
    source = """\
def f(cond):
    __guard   cond   else:
        return
    use()
"""
    expected = """\
def f(cond):
    __guard cond else:
        return
    use()
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_comptime_guard_condition():
    source = """\
def f[cond: Bool]():
    comptime __guard not  cond else:
        return
    use()
"""
    expected = """\
def f[cond: Bool]():
    comptime __guard not cond else:
        return
    use()
"""
    assert_mojo_format(source, expected)


def test_python_mode_keeps_case_indent():
    source = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    expected = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    assert_format(
        source,
        expected,
        mblack.Mode(target_versions={mblack.TargetVersion.PY310}),
        minimum_version=(3, 10),
    )


def test_mojo_formats_var_ref_patterns_in_kwargs():
    source = """\
def classify_point(p: Point) -> String:
    match p:
    case Point(x=var x, y=0):
        return String("x-axis:", x)
    case Point(x=0, y=ref y):
        return String("y-axis:", y)
    case Point(var x, 0):
        return String("x:", x)
    case (var x, var y):
        return String(x, y)
    case _:
        return "other"
"""
    expected = """\
def classify_point(p: Point) -> String:
    match p:
        case Point(x=var x, y=0):
            return String("x-axis:", x)
        case Point(x=0, y=ref y):
            return String("y-axis:", y)
        case Point(var x, 0):
            return String("x:", x)
        case (var x, var y):
            return String(x, y)
        case _:
            return "other"
"""
    assert_mojo_format(source, expected)
