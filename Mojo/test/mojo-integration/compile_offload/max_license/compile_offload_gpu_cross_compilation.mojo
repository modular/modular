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

# COM: cross compilation should work for all platforms no matter whether
# there is a nvidia GPU or not.
# RUN: %mojo-build --target-accelerator="nvidia:sm_90a" %s -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

from max.gpu.host import DeviceContext


def hello():
    pass


# CHECK: Generated
# CHECK: by LLVM NVPTX
def main() raises:
    with DeviceContext() as ctx:
        var t1 = ctx.compile_function[hello]()
        print(t1._func_impl.asm)
