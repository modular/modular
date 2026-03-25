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

from std.gpu import thread_idx_uint as thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host.func_attribute import Attribute
from std.testing import assert_equal


def test_function_attributes() raises:
    def kernel(x: UnsafePointer[Int, MutAnyOrigin]):
        x[0] = Int(thread_idx.x)

    with DeviceContext() as ctx:
        var func = ctx.compile_function_experimental[kernel]()
        assert_equal(func.get_attribute(Attribute.LOCAL_SIZE_BYTES), 0)


def main() raises:
    test_function_attributes()
