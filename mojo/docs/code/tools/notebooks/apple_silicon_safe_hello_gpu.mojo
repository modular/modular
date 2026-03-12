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

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

comptime `✅`: Int32 = 1
comptime `❌`: Int32 = 0


def kernel(value: UnsafePointer[Scalar[DType.int32], MutAnyOrigin]):
    value[0] = `✅`


def main() raises:
    with DeviceContext() as ctx:
        # Build it
        var out = ctx.enqueue_create_buffer[DType.int32](1)
        out.enqueue_fill(`❌`)

        # Run it
        ctx.enqueue_function[kernel, kernel](out, grid_dim=1, block_dim=1)

        # Print it
        with out.map_to_host() as out_host:
            print("GPU responded:", "✅" if out_host[0] == `✅` else "❌")
