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

# COM: MOCO-2418, test that we do not impose wrong alignment for large vector.

from max.gpu.host import DeviceContext


def kernel[
    size: Int,
    dtype: DType,
](x: SIMD[dtype, size], val: Int32):
    print(x)


def main() raises:
    with DeviceContext() as ctx:
        comptime simd_size = 32
        var x = SIMD[.float32, simd_size](0.0)

        comptime kernel_func = kernel[simd_size, DType.float32]
        ctx.enqueue_function[kernel_func](x, Int32(10), grid_dim=1, block_dim=1)
        ctx.synchronize()


# CHECK: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
