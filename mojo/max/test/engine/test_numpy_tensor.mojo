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
# UNSUPPORTED: asan
# RUN: %mojo -debug-level full %s

from max.engine import EngineNumpyView
from python import Python, PythonObject
from testing import assert_equal


fn test_numpy_view() raises:
    var np = Python.import_module("numpy")

    var n1 = np.array(Python.list(1, 2, 3)).astype(np.float32)

    var n1_view = EngineNumpyView(n1)

    assert_equal(String(n1_view.spec()), "3xfloat32")

    _ = n1^


fn main() raises:
    test_numpy_view()
