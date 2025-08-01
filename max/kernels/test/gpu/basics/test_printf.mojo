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

from io.io import _printf
from gpu.host import DeviceContext


# CHECK-LABEL: == test_gpu_printf
fn test_gpu_printf() raises:
    print("== test_gpu_printf")

    #
    # Test that stdlib _printf works on GPU
    #

    fn do_print(x: Int, y: Float64):
        # CHECK: printf printed 98 123.456!
        _printf["printf printed %ld %g!\n"](x, y)
        # CHECK: printf printed more 0 1 2 3 4 5 6 7 8 9
        _printf[
            "printf printed more %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n"
        ](0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    with DeviceContext() as ctx:
        ctx.enqueue_function[do_print](
            Int(98), Float64(123.456), grid_dim=1, block_dim=1
        )
        # Ensure queued function finished before proceeding.
        ctx.synchronize()


fn main() raises:
    test_gpu_printf()
