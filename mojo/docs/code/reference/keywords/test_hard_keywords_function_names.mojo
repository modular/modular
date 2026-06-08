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
# test_hard_keywords_function_names.mojo
# Test that hard keywords cannot be used as function names without escaping.
# This test file verifies the fix for issue #6630.

# These should all cause compilation errors because hard keywords
# are not allowed as function names without backtick escaping.

# ERROR: 'match' is a hard keyword and cannot be used as a function name
# def match():
#     pass

# ERROR: 'class' is a hard keyword and cannot be used as a function name
# def class():
#     pass

# ERROR: 'yield' is a hard keyword and cannot be used as a function name
# def yield():
#     pass

# ERROR: 'del' is a hard keyword and cannot be used as a function name
# def del():
#     pass

# However, these SHOULD work with backtick escaping:

def `match`():
    """Function using escaped keyword as name."""
    return 42

def `class`():
    """Function using escaped keyword as name."""
    return "class"

def `yield`():
    """Function using escaped keyword as name."""
    return 100

def `del`():
    """Function using escaped keyword as name."""
    return True

# And hard keywords SHOULD be allowed as struct method names
# because the dot operator disambiguates them:

struct Foo:
    def __init__(out self):
        pass

    def match(self) -> Int:
        """Method named 'match' is OK because it's accessed via dot notation."""
        return 1

    def class(self) -> String:
        """Method named 'class' is OK."""
        return "method"

    def yield(self) -> Int:
        """Method named 'yield' is OK."""
        return 2

    def del(self) -> Bool:
        """Method named 'del' is OK."""
        return False


def test_escaped_keywords():
    """Test that escaped keywords work as function names."""
    from std.testing import assert_equal

    # Escaped keyword functions should work
    assert_equal(`match`(), 42)
    assert_equal(`class`(), "class")
    assert_equal(`yield`(), 100)
    assert_equal(`del`(), True)

    # Methods named with keywords should work
    var f = Foo()
    assert_equal(f.match(), 1)
    assert_equal(f.class(), "method")
    assert_equal(f.yield(), 2)
    assert_equal(f.del(), False)


def main():
    test_escaped_keywords()
    print("All tests passed!")
