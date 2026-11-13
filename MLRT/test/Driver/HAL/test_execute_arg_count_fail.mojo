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
from max._driver.max_license.hal.plugins import FunctionHandle
from max._driver.max_license.hal.queue import Queue
from max.gpu.host import GridBlockLaunchConfig


def scale(x: Float32, result: MutPointer[Float32, MutUntrackedOrigin]):
    result[] = 2 * x


def launch[spec: DeviceSpec](queue: Queue[spec], func: FunctionHandle) raises:
    # CHECK: the kernel takes 2 arguments but the launch passes 1
    queue.execute[scale](
        Float32(1),
        function=func,
        execution_config=GridBlockLaunchConfig(grid_dim=1, block_dim=1),
    )


def main() raises:
    var driver = Driver.create("")
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()
    launch(queue[], ctx[].compile_function_inline[scale]())
