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

from max.gpu.host import DeviceContext
from layout import Coord, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.slice import sliced_add


def test_sliced_add[
    dtype: DType,
    rows: Int,
    cols: Int,
    batch_end_idx: Int,
](ctx: DeviceContext) raises:
    """Test the sliced_add_ragged kernel."""
    assert (
        batch_end_idx <= rows
    ), "batch_end_idx must be less than or equal to rows"

    # Create managed buffers and host views.
    var layout = row_major((rows, cols))
    var a = HostDeviceTileTensor[dtype](layout, ctx)
    var b = HostDeviceTileTensor[dtype](layout, ctx)
    var c = HostDeviceTileTensor[dtype](layout, ctx)
    var a_host = a.host_tensor()
    var b_host = b.host_tensor()
    var c_host = c.host_tensor()

    # Initialize with known patterns
    # a: all ones, b: all twos, c: zeros
    for i in range(rows):
        for j in range(cols):
            var idx = a_host.layout(Coord(i, j))
            a_host.raw_store(idx, 1.0)
            b_host.raw_store(idx, 2.0)
            c_host.raw_store(idx, 0.0)

    # Keep lora_end_idx on host; sliced_add reads this scalar on host.
    var lora_end_idx = HostDeviceTileTensor[.int64](row_major(1))
    var lora_end_idx_host = lora_end_idx.host_tensor()
    lora_end_idx_host.raw_store(0, Int64(batch_end_idx))

    a.to_device()
    b.to_device()
    c.to_device()
    var a_device_tensor = a.device_tensor()
    var b_device_tensor = b.device_tensor()
    var c_device_tensor = c.device_tensor()

    # Execute sliced_add directly
    sliced_add[target="gpu"](
        c_device_tensor,
        a_device_tensor,
        b_device_tensor,
        lora_end_idx_host,
        ctx,
    )

    c.to_host()

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

            var idx = c_host.layout(Coord(i, j))
            var actual = c_host.raw_load(idx)
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


def test_sliced_add_boundary_cases(ctx: DeviceContext) raises:
    # Test case 1: batch_end_idx = 0 (no addition, all copy)
    test_sliced_add[.float32, 4, 8, 0](ctx)

    # Test case 2: batch_end_idx = rows (all addition)
    test_sliced_add[.float32, 4, 8, 4](ctx)

    # Test case 3: batch_end_idx in middle
    test_sliced_add[.float32, 8, 16, 4](ctx)

    # Test case 4: Single row with addition
    test_sliced_add[.float32, 1, 8, 1](ctx)

    # Test case 5: Larger tensor
    test_sliced_add[.float32, 128, 64, 64](ctx)


def test_sliced_add_dtypes(ctx: DeviceContext) raises:
    test_sliced_add[.float32, 16, 32, 8](ctx)
    test_sliced_add[.float16, 16, 32, 8](ctx)
    test_sliced_add[.bfloat16, 16, 32, 8](ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_sliced_add_boundary_cases(ctx)
        test_sliced_add_dtypes(ctx)
