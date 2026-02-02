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

from gpu.host import DeviceContext
from layout._coord import Coord
from layout._layout import row_major
from layout._tile_tensor import TileTensor
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
    var shape = IndexList[2](rows, cols)
    var layout = row_major(Coord(shape))
    var size = rows * cols
    var a_host_ptr = alloc[Scalar[dtype]](size)
    var a_host = TileTensor(a_host_ptr, layout)
    var b_host_ptr = alloc[Scalar[dtype]](size)
    var b_host = TileTensor(b_host_ptr, layout)
    var c_host_ptr = alloc[Scalar[dtype]](size)
    var c_host = TileTensor(c_host_ptr, layout)

    # Initialize with known patterns
    # a: all ones, b: all twos, c: zeros
    for i in range(rows):
        for j in range(cols):
            var idx = a_host.layout(Coord(IndexList[2](i, j)))
            a_host.ptr[idx] = 1.0
            b_host.ptr[idx] = 2.0
            c_host.ptr[idx] = 0.0

    # Create lora_end_idx buffer (kept on host since sliced_add reads it on host)
    var lora_end_idx_host_ptr = alloc[Int64](1)
    var lora_end_idx_host = TileTensor(
        lora_end_idx_host_ptr, row_major(Coord(IndexList[1](1)))
    )
    lora_end_idx_host.ptr[0] = Int64(batch_end_idx)

    # Copy to device (lora_end_idx stays on host)
    var a_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(a_device, a_host_ptr)
    var a_device_tensor = TileTensor(a_device.unsafe_ptr(), layout)
    var b_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(b_device, b_host_ptr)
    var b_device_tensor = TileTensor(b_device.unsafe_ptr(), layout)
    var c_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(c_device, c_host_ptr)
    var c_device_tensor = TileTensor(c_device.unsafe_ptr(), layout)

    # Execute sliced_add directly
    sliced_add[target="gpu"](
        c_device_tensor,
        a_device_tensor,
        b_device_tensor,
        lora_end_idx_host,
        Optional(ctx),
    )

    # Copy result back to host
    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_device)
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

            var idx = c_host.layout(Coord(IndexList[2](i, j)))
            var actual = c_host.ptr[idx]
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

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    lora_end_idx_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^


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
