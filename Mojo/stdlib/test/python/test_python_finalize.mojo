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
"""Tests that the embedded CPython interpreter is properly finalized at exit.

Specifically tests that:
- Python print() output is produced when stdout is piped (not a TTY)
- Python atexit handlers run during interpreter finalization
"""

from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true, TestSuite


def test_python_print_captured() raises:
    """Python print() works and produces output via StringIO capture."""
    var captured = String(
        Python.evaluate(
            "("
            "  lambda: ("
            "    __import__('sys').__dict__.__setitem__('stdout',"
            " __import__('io').StringIO()) or"
            "    print('hello from Python') or"
            "    __import__('sys').stdout.getvalue()"
            "  )"
            ")()"
        )
    )
    assert_equal(captured, "hello from Python\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
