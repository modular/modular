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
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    InitializationType,
    initialize,
)
from nn.slice import sliced_add

from utils import IndexList


fn test_sliced_add[
    dtype: DType,
    rows: Int,
    cols: Int,
    batch_end_idx: Int,
](ctx: DeviceContext) raises:
    """Test the sliced_add_ragged kernel."""
    debug_assert(
        batch_end_idx <= rows,
        "batch_end_idx must be less than or equal to rows",
    )

    # Create host buffers
    var a_host = HostNDBuffer[dtype, 2](IndexList[2](rows, cols))
    var b_host = HostNDBuffer[dtype, 2](IndexList[2](rows, cols))
    var c_host = HostNDBuffer[dtype, 2](IndexList[2](rows, cols))

    # Initialize with known patterns
    # a: all ones
    initialize(a_host.tensor, InitializationType.one)
    # b: all twos
    for i in range(rows):
        for j in range(cols):
            b_host.tensor[i, j] = 2.0
    # c: zeros (will be overwritten)
    initialize(c_host.tensor, InitializationType.zero)

    # Create lora_end_idx buffer (kept on host since sliced_add reads it on host)
    var lora_end_idx_host = HostNDBuffer[DType.int64, 1](IndexList[1](1))
    lora_end_idx_host.tensor[0] = Int64(batch_end_idx)

    # Copy to device (lora_end_idx stays on host)
    var a_device = a_host.copy_to_device(ctx)
    var b_device = b_host.copy_to_device(ctx)
    var c_device = c_host.copy_to_device(ctx)

    # Execute sliced_add directly
    sliced_add[target="gpu"](
        c_device.to_layout_tensor(),
        a_device.to_layout_tensor(),
        b_device.to_layout_tensor(),
        lora_end_idx_host.to_layout_tensor(),
        Optional(ctx),
    )

    # Copy result back to host
    ctx.synchronize()
    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.synchronize()

    # Verify results
    for i in range(rows):
        for j in range(cols):
            var expected: Scalar[dtype]
            if i < batch_end_idx:
                # Should be a + b = 1 + 2 = 3
                expected = 3.0
            else:
                # Should be just a = 1
                expected = 1.0

            var actual = c_host.tensor[i, j]
            if actual != expected:
                raise Error(
                    "Mismatch at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: expected "
                    + String(expected)
                    + ", got "
                    + String(actual)
                )


fn test_sliced_add_boundary_cases(ctx: DeviceContext) raises:
    # Test case 1: batch_end_idx = 0 (no addition, all copy)
    test_sliced_add[DType.float32, 4, 8, 0](ctx)

    # Test case 2: batch_end_idx = rows (all addition)
    test_sliced_add[DType.float32, 4, 8, 4](ctx)

    # Test case 3: batch_end_idx in middle
    test_sliced_add[DType.float32, 8, 16, 4](ctx)

    # Test case 4: Single row with addition
    test_sliced_add[DType.float32, 1, 8, 1](ctx)

    # Test case 5: Larger tensor
    test_sliced_add[DType.float32, 128, 64, 64](ctx)


fn test_sliced_add_dtypes(ctx: DeviceContext) raises:
    test_sliced_add[DType.float32, 16, 32, 8](ctx)
    test_sliced_add[DType.float16, 16, 32, 8](ctx)
    test_sliced_add[DType.bfloat16, 16, 32, 8](ctx)


def main():
    with DeviceContext() as ctx:
        test_sliced_add_boundary_cases(ctx)
        test_sliced_add_dtypes(ctx)
