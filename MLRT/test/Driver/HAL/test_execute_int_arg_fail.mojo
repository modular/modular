# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo-no-debug --target-accelerator=nvidia:sm80 %s 2>&1 | FileCheck %s

from max._driver import Driver
from max._driver.max_license.hal.device import DeviceSpec
from max._driver.max_license.hal.queue import Queue
from max.gpu.host import GridBlockLaunchConfig


def fill(result: MutPointer[Float32, MutUntrackedOrigin], n: Int):
    pass


def launch[spec: DeviceSpec](queue: Queue[spec]) raises:
    # `Int` is 8 bytes on the host but may be 4 on the device.
    # CHECK: argument #1 is an `Int` or `UInt`, whose width differs between host and device; use a fixed-width type such as `Int32` or `Int64`
    queue.execute[fill](
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=8),
        Int(4),
        execution_config=GridBlockLaunchConfig(grid_dim=1, block_dim=1),
    )


def main() raises:
    var driver = Driver.create("")
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()
    launch(queue[])
