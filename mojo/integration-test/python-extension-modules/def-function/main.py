# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys
import unittest
from typing import Callable

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module as def_function


class TestPythonModuleBuilderDefFunction(unittest.TestCase):
    def _test_takes_zero_returns(self, fut: Callable[[], str]):
        setattr(sys.modules[__name__], "s", "just a python string")
        result = fut()
        self.assertEqual(result, "just another python string")

    def _test_takes_one_returns(self, fut: Callable[[str], str]):
        result = fut("foo")
        self.assertEqual(result, "foo")

    def _test_takes_two_returns(self, fut: Callable[[str, str], str]):
        result = fut("foo", "bar")
        self.assertEqual(result, "foobar")

    def _test_takes_three_returns(self, fut: Callable[[str, str, str], str]):
        result = fut("foo", "bar", "baz")
        self.assertEqual(result, "foobarbaz")

    def test_takes_zero_returns(self):
        self._test_takes_zero_returns(def_function.takes_zero_returns)

    def test_takes_one_returns(self):
        self._test_takes_one_returns(def_function.takes_one_returns)

    def test_takes_two_returns(self):
        self._test_takes_two_returns(def_function.takes_two_returns)

    def test_takes_three_returns(self):
        self._test_takes_three_returns(def_function.takes_three_returns)

    def test_takes_zero_raises_returns(self):
        setattr(sys.modules[__name__], "s", "a special python string")
        with self.assertRaises(Exception) as cm:
            def_function.takes_zero_raises_returns()
        self.assertEqual(
            cm.exception.args, ("`s` must be 'just a python string'",)
        )

        self._test_takes_zero_returns(def_function.takes_zero_raises_returns)

    def test_takes_one_raises_returns(self):
        with self.assertRaises(Exception) as cm:
            def_function.takes_one_raises_returns("quux")
        self.assertEqual(cm.exception.args, ("input must be 'foo'",))

        self._test_takes_one_returns(def_function.takes_one_raises_returns)

    def test_takes_two_raises_returns(self):
        with self.assertRaises(Exception) as cm:
            def_function.takes_two_raises_returns("quux", "bar")
        self.assertEqual(cm.exception.args, ("first input must be 'foo'",))

        self._test_takes_two_returns(def_function.takes_two_raises_returns)

    def test_takes_three_raises_returns(self):
        with self.assertRaises(Exception) as cm:
            def_function.takes_three_raises_returns("quux", "bar", "baz")
        self.assertEqual(cm.exception.args, ("first input must be 'foo'",))

        self._test_takes_three_returns(def_function.takes_three_raises_returns)


if __name__ == "__main__":
    unittest.main()
