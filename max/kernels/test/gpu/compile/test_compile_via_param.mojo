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
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug -D DUMP_GPU_ASM=True %s | FileCheck %s
# RUN: rm -fr %tmp-dir/test_compile_via_param/
# RUN: mkdir -p %tmp-dir/test_compile_via_param/
# RUN: %mojo-no-debug -D DUMP_GPU_ASM=%tmp-dir/test_compile_via_param/test_compile_via_param.ptx %s
# RUN: cat %tmp-dir/test_compile_via_param/test_compile_via_param.ptx | FileCheck %s
# RUN: rm -fr %tmp-dir/test_compile_via_param/

from gpu import thread_idx
from gpu.host import DeviceContext


def test_compile_function():
    print("== test_compile_function")

    fn kernel(x: UnsafePointer[Int]):
        x[0] = thread_idx.x

    # CHECK: tid.x

    with DeviceContext() as ctx:
        _ = ctx.compile_function[kernel]()


def main():
    test_compile_function()
