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
from max._driver.max_license.hal.execution_config._nvidia.a100 import (
    A100ExecutionConfiguration,
)
from max._driver.max_license.hal.queue import Queue


def scale(x: Float32, result: MutPointer[Float32, MutUntrackedOrigin]):
    result[] = 2 * x


def launch[spec: DeviceSpec](queue: Queue[spec]) raises:
    # A vendor config can hold constant memory mappings, which the launch
    # would drop.
    # CHECK: HAL `Queue.execute` does not support `constant_memory` mappings.
    queue.execute[scale](
        Float32(1),
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=8),
        execution_config=A100ExecutionConfiguration(grid_dim=1, block_dim=1),
    )


def main() raises:
    var driver = Driver.create("")
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()
    launch(queue[])
