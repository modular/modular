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

"""Test FP8 E4M3FN to E4M3FNUZ conversion kernel."""

from buffer import NDBuffer
from gpu.host import DeviceContext
from linalg.fp8_quantization import convert_e4m3fn_to_e4m3fnuz
from testing import assert_equal
from memory import bitcast
from internal_utils import DeviceNDBuffer, HostNDBuffer
from buffer.dimlist import DimList


# CHECK-LABEL: test_convert_e4m3fn_to_e4m3fnuz_basic
fn test_convert_e4m3fn_to_e4m3fnuz_basic() raises:
    print("test_convert_e4m3fn_to_e4m3fnuz_basic")
    var ctx = DeviceContext()

    # Test with 5 values: 4 regular values + 1 special -128 bit pattern
    var host_in = HostNDBuffer[DType.float8_e4m3fn, 2](DimList(1, 5))
    var host_out = HostNDBuffer[DType.float8_e4m3fnuz, 2](DimList(1, 5))

    # Regular values - these should pass through unchanged (same bits, different interpretation)
    host_in.tensor.data[0] = Float8_e4m3fn(1.0)
    host_in.tensor.data[1] = Float8_e4m3fn(2.0)
    host_in.tensor.data[2] = Float8_e4m3fn(-1.0)
    host_in.tensor.data[3] = Float8_e4m3fn(0.0)
    host_in.tensor.data[4] = Float8_e4m3fn(
        -0.0
    )  # Special 0x80 bit pattern - this should be converted to 0.0

    var device_in = DeviceNDBuffer[DType.float8_e4m3fn, 2](
        DimList(1, 5), ctx=ctx
    )
    var device_out = DeviceNDBuffer[DType.float8_e4m3fnuz, 2](
        DimList(1, 5), ctx=ctx
    )

    ctx.enqueue_copy(device_in.buffer, host_in.tensor.data)

    convert_e4m3fn_to_e4m3fnuz(device_in.tensor, device_out.tensor, ctx)
    ctx.enqueue_copy(host_out.tensor.data, device_out.buffer)
    ctx.synchronize()

    # Verify results: regular values should be unchanged in bits, -128 should become 0
    print("Input values:")
    for i in range(5):
        print("  [", i, "]:", host_in.tensor.data[i])
    # CHECK: Input values:
    # CHECK:   [ 0 ]: 1.0
    # CHECK:   [ 1 ]: 2.0
    # CHECK:   [ 2 ]: -1.0
    # CHECK:   [ 3 ]: 0.0
    # CHECK:   [ 4 ]: -0.0

    print("Output values:")
    for i in range(5):
        print("  [", i, "]:", host_out.tensor.data[i])
    # CHECK: Output values:
    # CHECK:   [ 0 ]: 0.5
    # CHECK:   [ 1 ]: 1.0
    # CHECK:   [ 2 ]: -0.5
    # CHECK:   [ 3 ]: 0.0
    # CHECK:   [ 4 ]: 0.0

    print("Conversion verification: same bits, different FP8 interpretation")
    # CHECK: Conversion verification: same bits, different FP8 interpretation


fn main() raises:
    test_convert_e4m3fn_to_e4m3fnuz_basic()
