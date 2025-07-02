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

# Put the current directory (containing .so) on the Python module lookup path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module


class TestMojoPairNonTrivialInit(unittest.TestCase):
    """Test non-trivial initialization of MojoPair from Python."""

    def test_non_trivial_initialization(self):
        """Test creating MojoPair with custom values MojoPair(42, 10)."""
        pair = mojo_module.MojoPair(42, 10)

        # Verify the values were set correctly
        self.assertEqual(pair.get_first(), 42)
        self.assertEqual(pair.get_second(), 10)

    def test_swap_method(self):
        """Test the swap method works correctly."""
        pair = mojo_module.MojoPair(42, 10)

        # Before swap
        self.assertEqual(pair.get_first(), 42)
        self.assertEqual(pair.get_second(), 10)

        # Swap and verify
        result = pair.swap()
        self.assertEqual(result, pair)  # Should return self
        self.assertEqual(pair.get_first(), 10)
        self.assertEqual(pair.get_second(), 42)

    def test_error_handling(self):
        """Test error handling for invalid initialization arguments."""
        # These should raise appropriate errors
        with self.assertRaises(Exception):
            mojo_module.MojoPair("not_a_number", 10)

        with self.assertRaises(Exception):
            mojo_module.MojoPair(42, "not_a_number")


if __name__ == "__main__":
    unittest.main()
