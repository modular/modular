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
# RUN: %mojo-no-debug %s

from gpu import thread_idx
from gpu.host import DeviceContext
from gpu.host.func_attribute import Attribute
from memory import UnsafePointer, stack_allocation
from testing import *


# CHECK-LABEL: test_function_attributes
def test_function_attributes():
    fn kernel(x: UnsafePointer[Int]):
        x[0] = thread_idx.x

    with DeviceContext() as ctx:
        var func = ctx.compile_function[kernel]()
        assert_equal(func.get_attribute(Attribute.CONST_SIZE_BYTES), 0)


def main():
    test_function_attributes()
