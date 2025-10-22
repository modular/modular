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

from gpu.host import DeviceContext
from gpu.mma import mma
from testing import assert_equal


fn test_mma_fp16_kernel(c_ptr: UnsafePointer[Float32]):
    """Simple FP16×FP16+FP32→FP32 MMA test kernel.

    This test performs a basic matrix multiply-accumulate operation using
    FP16 inputs and FP32 accumulator. On different GPU architectures, this
    operation maps to:
    - NVIDIA: Uses tensor core wmma or mma.sync instructions
    - AMD CDNA: Uses mfma instructions
    - AMD RDNA3+: Uses v_wmma_f32_16x16x16_f16 instructions
    - AMD RDNA1/2: Falls back to scalar operations (no WMMA support)

    IMPORTANT - RDNA3 WMMA Bug (Fixed October 2025):
    RDNA3 WMMA instructions were broken in all LLVM versions 15.0.0-22.0.0git
    for compute kernels (amdgpu_kernel calling convention). Graphics shaders
    worked, but HIP/ROCm compute kernels failed with "Cannot select intrinsic".

    Mojo 25.5.0's LLVM confirmed to have this bug - using `mojo build -o llvm`
    fails during IR generation, preventing workarounds via external llc.

    LLVM Fix Status:
    Submitted upstream: https://github.com/llvm/llvm-project/pull/164036
    Expected path: Modular will backport fix to Mojo's LLVM

    This test requires either:
      1. LLVM 23+ with upstreamed fix (after PR merges), OR
      2. Mojo's LLVM with backported fix (expected), OR
      3. ROCm's LLVM (TheRock) which already has the fix

    See RDNA3_WMMA_PROJECT_STATUS.md for complete details.

    The test validates that the mma() intrinsic correctly lowers to
    appropriate hardware instructions for the target platform.

    Args:
        c_ptr: Output buffer for results (4 FP32 values).
    """
    var a_reg = SIMD[DType.float16, 4](1.0, 2.0, 3.0, 4.0)
    var b_reg = SIMD[DType.float16, 4](1.0, 1.0, 1.0, 1.0)
    var c_reg = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var d_reg = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

    mma(d_reg, a_reg, b_reg, c_reg)

    c_ptr[0] = d_reg[0]
    c_ptr[1] = d_reg[1]
    c_ptr[2] = d_reg[2]
    c_ptr[3] = d_reg[3]


def main():
    """Test FP16 matrix multiply-accumulate operation."""
    with DeviceContext() as ctx:
        var c_device = ctx.enqueue_create_buffer[DType.float32](4)
        var c_host = UnsafePointer[Float32].alloc(4)

        for i in range(4):
            c_host[i] = -1.0

        ctx.enqueue_copy(c_device, c_host)

        alias kernel = test_mma_fp16_kernel

        ctx.enqueue_function_checked[kernel, kernel](
            c_device,
            grid_dim=1,
            block_dim=64,
        )

        ctx.enqueue_copy(c_host, c_device)
        ctx.synchronize()

        for i in range(4):
            assert_equal(c_host[i] != -1.0, True)

        _ = c_device
        c_host.free()
