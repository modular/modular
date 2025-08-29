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

import gpu
from gpu.host import DeviceContext, DeviceMulticastBuffer
from gpu.memory import multimem_ld_reduce, ReduceOp, Scope, Consistency
from memory.pointer import _GPUAddressSpace
from sys.info import simd_width_of
from testing import assert_true


fn multimem_reduction_kernel[
    dtype: DType,
](
    multicast_ptr: UnsafePointer[
        Scalar[dtype], address_space = _GPUAddressSpace.GLOBAL
    ],
    result_ptr: UnsafePointer[
        Scalar[dtype], address_space = _GPUAddressSpace.GLOBAL
    ],
    size: Int,
):
    """Kernel that uses multimem_ld_reduce wrapper for cross-GPU reduction."""
    alias simd_width = simd_width_of[dtype]()

    var tid = gpu.thread_idx.x + gpu.block_idx.x * gpu.block_dim.x
    if simd_width * tid >= size:
        return

    # Each thread processes simd_width elements using the wrapper function
    var element_addr = multicast_ptr + simd_width * tid

    var results = multimem_ld_reduce[
        dtype,
        simd_width=simd_width,
        reduction = ReduceOp.ADD,
        scope = Scope.GPU,
        consistency = Consistency.RELAXED,
        accum_type = DType.float32,
    ](element_addr)

    # Store the reduced results for simd_width elements
    for i in range(simd_width):
        if simd_width * tid + i < size:
            result_ptr[simd_width * tid + i] = results[i]


fn test_multicast_reduction[
    dtype: DType,
    ngpus: Int,
](contexts: List[DeviceContext], test_size: Int = 512) raises:
    """Test multicast ADD reduction operations across multiple GPUs using optimal simd_width.
    """
    constrained[ngpus in (2, 4), "ngpus must be 2 or 4"]()

    print(
        "=== Testing",
        ngpus,
        "GPU multicast ADD reduction with",
        String(dtype),
        "===",
    )

    # Validate parameters
    if len(contexts) != ngpus:
        raise Error("Number of contexts must match ngpus parameter")

    # Check if all devices support multicast
    for i in range(ngpus):
        if not contexts[i].supports_multicast():
            raise Error("Device " + String(i) + " does not support multicast")

    # Create multicast buffer - same size for all GPUs to access same virtual addresses
    var input_multicast_buf = DeviceMulticastBuffer[dtype](contexts, test_size)

    # Create regular buffer for results on first GPU
    var result_buffer = contexts[0].enqueue_create_buffer[dtype](test_size)

    # Initialize input data - each GPU writes to the same indices with different values
    print("Initializing test data...")
    for gpu_id in range(ngpus):
        var dev_buf = input_multicast_buf.unicast_buffer_for(contexts[gpu_id])

        with dev_buf.map_to_host() as host_buf:
            for j in range(test_size):
                # Each GPU writes different values to index j
                # Use small values within bfloat16's precision range (max ~127)
                if gpu_id == 0:
                    host_buf[j] = Scalar[dtype](j % 64)  # 0 to 63
                elif gpu_id == 1:
                    host_buf[j] = Scalar[dtype](63 - (j % 64))  # 63 to 0
                elif gpu_id == 2:
                    host_buf[j] = Scalar[dtype](j % 64)  # 0 to 63
                else:  # gpu_id == 3
                    host_buf[j] = Scalar[dtype](63 - (j % 64))  # 63 to 0

    # Initialize result buffer to zeros
    with result_buffer.map_to_host() as host_buf:
        for j in range(test_size):
            host_buf[j] = Scalar[dtype](0)

    # Launch kernel on first GPU
    print("Launching kernel...")
    # Get the MULTICAST address (shared across all GPUs)
    var input_multicast_ptr = input_multicast_buf.multicast_buffer_for(
        contexts[0]
    ).unsafe_ptr()

    var threads_needed = (
        test_size + 255
    ) // 256  # Use block size as approximation
    contexts[0].enqueue_function[
        multimem_reduction_kernel[dtype], dump_asm=False
    ](
        input_multicast_ptr,
        result_buffer.unsafe_ptr(),
        test_size,
        grid_dim=((threads_needed + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )

    # Synchronize all contexts
    for i in range(ngpus):
        contexts[i].synchronize()

    # Verify results
    print("Verifying results...")
    with result_buffer.map_to_host() as host_buf:
        var errors = 0
        for i in range(test_size):
            var actual = host_buf[i]

            # Calculate expected result for ADD operation at index i
            var expected: Scalar[dtype]
            if ngpus == 2:
                # For 2 GPUs: GPU 0: i%64, GPU 1: 63-(i%64)
                # Total: (i%64) + (63-(i%64)) = 63
                expected = Scalar[dtype](63)
            else:  # ngpus == 4
                # For 4 GPUs: GPU 0: i%64, GPU 1: 63-(i%64), GPU 2: i%64, GPU 3: 63-(i%64)
                # Total: (i%64) + (63-(i%64)) + (i%64) + (63-(i%64)) = 2*(i%64) + 2*(63-(i%64)) = 126
                expected = Scalar[dtype](126)

            if abs(actual - expected) > 1e-5:
                if errors < 10:  # Only print first 10 errors
                    print(
                        "Error at index",
                        i,
                        ": expected",
                        expected,
                        ", got",
                        actual,
                    )
                errors += 1

        if errors == 0:
            print("✓ All", test_size, "results correct!")
        else:
            print("✗", errors, "errors found")
            raise Error("Result validation failed")


def main():
    print("=== Multicast Memory Reduction Tests ===")

    # Check device availability and multicast support
    var num_gpus = DeviceContext.number_of_devices(api="gpu")
    print("Available GPUs:", num_gpus)

    # Require at least 2 GPUs for multicast tests
    assert_true(num_gpus >= 2, "must have at least 2 GPUs for multicast tests")

    with DeviceContext() as ctx0:
        if not ctx0.supports_multicast():
            print("Multicast memory not supported")
            return

        try:
            # Determine how many GPUs to test with (2 or 4)
            var test_gpus = 4 if num_gpus >= 4 else 2
            print("\n---", test_gpus, "GPU Tests ---")

            var contexts = List[DeviceContext]()
            for i in range(test_gpus):
                contexts.append(DeviceContext(device_id=i))

            if test_gpus == 2:
                test_multicast_reduction[DType.float32, 2](contexts)
                test_multicast_reduction[DType.bfloat16, 2](contexts)
            else:
                test_multicast_reduction[DType.float32, 4](contexts)
                test_multicast_reduction[DType.bfloat16, 4](contexts)

            print("\n=== All Tests Completed Successfully! ===")

        except e:
            print("Test failed with error:", e)
            raise e
