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

from tests.util import assert_mojo_format


def test_functype_no_space_before_parens():
    """No space between metaparams and parens in function types."""
    source = "def takes_bar[bar: def[x: Int]()](): pass"
    expected = "def takes_bar[bar: def[x: Int]()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_space_before_parens_with_args():
    """No space between metaparams and parens in function types with args."""
    source = "def takes_bar[bar: def[x: Int](y: Int)](): pass"
    expected = "def takes_bar[bar: def[x: Int](y: Int)]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_no_metaparams():
    """Function types without metaparams should also not have extra space."""
    source = "def takes_bar[bar: def()](): pass"
    expected = "def takes_bar[bar: def()]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_functype_with_return_type():
    """Function types with return types should format correctly."""
    source = "def takes_bar[bar: def[x: Int]() -> Int](): pass"
    expected = "def takes_bar[bar: def[x: Int]() -> Int]():\n    pass\n"
    assert_mojo_format(source, expected)


def test_named_effect_no_space_before_parens():
    """No space between effect name and parens (e.g., abi("C"))."""
    source = 'def foo() abi ("C"):\n    pass\n'
    expected = 'def foo() abi("C"):\n    pass\n'
    assert_mojo_format(source, expected)


def test_named_effect_no_space_already_correct():
    """abi("C") without space should remain unchanged."""
    source = 'def foo() abi("C"):\n    pass\n'
    expected = 'def foo() abi("C"):\n    pass\n'
    assert_mojo_format(source, expected)


def test_accesses_clause():
    """Formats the experimental accesses clause."""
    source = (
        'def foo(x: Int) -> Int __accesses ('
        '"__unsafe_nested_origins_read_only"): pass\n'
    )
    expected = (
        'def foo(x: Int) -> Int __accesses('
        '"__unsafe_nested_origins_read_only"):\n'
        '    pass\n'
    )
    assert_mojo_format(source, expected)


def test_accesses_clause_on_function_type():
    """Formats accesses clauses on function type annotations."""
    source = (
        'def takes(f: def(Int) -> Int __accesses ('
        '"__unsafe_nested_origins_read_only")): pass\n'
    )
    expected = (
        'def takes(f: def(Int) -> Int __accesses('
        '"__unsafe_nested_origins_read_only")):\n'
        '    pass\n'
    )
    assert_mojo_format(source, expected)
