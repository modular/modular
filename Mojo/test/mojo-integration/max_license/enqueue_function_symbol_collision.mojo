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

# REQUIRES: NVIDIA-GPU || AMD-GPU
# RUN: %mojo %s

# Regression test for MOCO-3056: enqueue_function should not cause
# symbol collision when the same kernel template is instantiated with different
# type parameters.
#
# This tests the scenarios described in the issue:
# - Multiple dtype configurations
# - Multiple kernel variants with different parameters
# - Same kernel template with different tile sizes/configurations

from max.gpu.host import DeviceContext


def generic_kernel[T: DType]():
    pass


def configurable_kernel[T: DType, tile_size: Int]():
    pass


def simd_kernel[simd_width: Int, dtype: DType]():
    pass


def main() raises:
    with DeviceContext() as ctx:
        # Multiple dtype instantiations
        ctx.enqueue_function[generic_kernel[.float32]](grid_dim=1, block_dim=32)
        ctx.enqueue_function[generic_kernel[.float16]](grid_dim=1, block_dim=32)
        ctx.enqueue_function[generic_kernel[.int32]](grid_dim=1, block_dim=32)

        # Same dtype but different tile sizes
        ctx.enqueue_function[configurable_kernel[.float32, 32]](
            grid_dim=1, block_dim=32
        )
        ctx.enqueue_function[configurable_kernel[.float32, 64]](
            grid_dim=1, block_dim=32
        )
        ctx.enqueue_function[configurable_kernel[.float32, 128]](
            grid_dim=1, block_dim=32
        )

        # Different dtypes and same tile size
        ctx.enqueue_function[configurable_kernel[.float16, 32]](
            grid_dim=1, block_dim=32
        )

        # Varied SIMD widths and dtypes
        ctx.enqueue_function[simd_kernel[4, DType.float32]](
            grid_dim=1, block_dim=32
        )
        ctx.enqueue_function[simd_kernel[8, DType.float32]](
            grid_dim=1, block_dim=32
        )
        ctx.enqueue_function[simd_kernel[4, DType.float16]](
            grid_dim=1, block_dim=32
        )

        ctx.synchronize()
