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
