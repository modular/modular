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

# REQUIRES: NVIDIA-GPU
# RUN: %mojo %s | FileCheck %s

# COM: test GPU kernel file name starts with a number.
# PTX compiler cannot parse function name starts with a number,
# but we attach file name to GPU functions which makes function name starts with a number.
# The compiler will insert `_` prefix to workaround the limitation.

from max.gpu import thread_idx
from max.gpu.host import DeviceContext


def main() raises:
    def printing_kernel():
        print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

    var ctx = DeviceContext()

    ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
    ctx.synchronize()


# CHECK: GPU thread: [ 0 0 0 ]
# CHECK: GPU thread: [ 1 0 0 ]
# CHECK: GPU thread: [ 2 0 0 ]
# CHECK: GPU thread: [ 3 0 0 ]
