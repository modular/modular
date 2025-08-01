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
from testing import assert_equal


def test_enqueue_fill_host_buffer(ctx: DeviceContext):
    var host_buffer = ctx.enqueue_create_host_buffer[DType.float64](8)
    _ = host_buffer.enqueue_fill(0.1)
    ctx.synchronize()

    # Verify all elements are filled with 0.1
    for i in range(8):
        assert_equal(
            host_buffer[i],
            0.1,
            String("host_buffer[") + String(i) + "] should be 0.1",
        )


def main():
    with DeviceContext() as ctx:
        test_enqueue_fill_host_buffer(ctx)
