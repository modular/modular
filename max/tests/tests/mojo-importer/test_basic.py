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

import sys
import unittest

import mojo.importer  # noqa
import numpy as np

# Put the current directory (containing mojo_module.mojo) on the Python module
# lookup path.
sys.path.insert(0, "")


# Imports from 'mojo_module.mojo'
import mojo_module  # type: ignore


class TestMojoPythonInterop(unittest.TestCase):
    def test_pyinit(self) -> None:
        self.assertTrue(mojo_module)

    def test_plus_one(self) -> None:
        self.assertEqual(mojo_module.plus_one(5), 6)

    def test_parallel_wrapper(self) -> None:
        arr = np.random.uniform(size=(10000)).astype(np.float32)
        arr = mojo_module.parallel_wrapper(arr)
        assert np.allclose(arr[0], np.max(arr))


if __name__ == "__main__":
    unittest.main()
