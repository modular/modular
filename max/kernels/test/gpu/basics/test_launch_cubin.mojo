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

from pathlib import Path
from testing import assert_equal
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.host.device_context import enqueue_function_from_cubin
from gpu.host.dim import Dim

# =============================================================================
# CUDA Kernel (vector_add.cu):
# =============================================================================
#
# extern "C" {
#
# __global__ void vector_add(const float* a, const float* b, float* c, int size) {
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;
#
#     if (idx < size) {
#         c[idx] = a[idx] + b[idx];
#     }
# }
#
# }  // extern "C"
#
# =============================================================================
# Compile to cubin for B200 (SM 100):
#     nvcc -cubin -arch=sm_100 -o vector_add.cubin vector_add.cu
# =============================================================================


fn main() raises:
    """Load the vector_add.cubin file and launch the kernel."""
    # Path relative to workspace root for bazel runfiles
    var cubin_path = Path("max/kernels/test/gpu/basics/vector_add.cubin")

    # Read the cubin file as bytes
    var cubin_bytes = cubin_path.read_bytes()
    print("Cubin file size:", len(cubin_bytes), "bytes")

    # Create a device context using context manager
    with DeviceContext() as ctx:
        var size = 1024

        # Create host buffers and initialize with incrementing values
        var a_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var b_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()

        # Initialize a and b: a[i] = i, b[i] = i
        for i in range(size):
            a_host[i] = Float32(i)
            b_host[i] = Float32(i)

        # Allocate device buffers
        var a = ctx.enqueue_create_buffer[DType.float32](size)
        var b = ctx.enqueue_create_buffer[DType.float32](size)
        var c = ctx.enqueue_create_buffer[DType.float32](size)

        # Copy host data to device
        ctx.enqueue_copy(a, a_host)
        ctx.enqueue_copy(b, b_host)
        c.enqueue_fill(Float32(0.0))

        # Launch the external kernel: c = a + b
        enqueue_function_from_cubin(
            ctx,
            cubin_bytes,
            "vector_add_module",
            "vector_add",
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            c.unsafe_ptr(),
            Int32(size),
            grid_dim=Dim((size + 255) // 256, 1, 1),
            block_dim=Dim(256, 1, 1),
        )

        # Create host buffer and copy result from device
        var c_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.enqueue_copy(c_host, c)
        ctx.synchronize()

        # Verify c[i] == a[i] + b[i] == i + i == 2*i
        for i in range(size):
            assert_equal(c_host[i], Float32(2 * i))

        print("SUCCESS: All", size, "values are correct (c[i] = 2*i)")
