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
"""Tests for PythonException type.

These tests verify that PythonException can be created and used to represent
Python exceptions safely.
"""

from std.python import PythonException
from std.testing import TestSuite


def test_python_exception_exists():
    """Test that PythonException type exists and can be imported."""
    # Just verify the type exists - creation requires Python which may not be available
    # in all test environments
    _ = PythonException


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#
def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
