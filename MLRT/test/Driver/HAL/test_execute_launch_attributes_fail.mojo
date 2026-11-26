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
# RUN: not %mojo-no-debug --target-accelerator=amdgpu:gfx950 %s 2>&1 | FileCheck %s

from max._driver import Driver
from max._driver.max_license.hal.device import DeviceSpec
from max._driver.max_license.hal.queue import Queue
from max.gpu.host import Dim, GridBlockExecutionConfig, LaunchAttribute
from max.gpu.host.nvidia import LaunchAttributeHolderExecutionConfig


struct AttributeConfig(
    GridBlockExecutionConfig, LaunchAttributeHolderExecutionConfig
):
    var attributes: List[LaunchAttribute]

    def __init__(out self):
        self.attributes = []

    def get_grid_dim(self) -> Dim:
        return Dim(1)

    def get_block_dim(self) -> Dim:
        return Dim(1)

    def get_launch_attributes[
        o: ImmOrigin
    ](ref[o] self) -> Span[LaunchAttribute, o]:
        return rebind[Span[LaunchAttribute, o]](Span(self.attributes))


def scale(x: Float32, result: MutPointer[Float32, MutUntrackedOrigin]):
    result[] = 2 * x


def launch[spec: DeviceSpec](queue: Queue[spec]) raises:
    # The AMD launcher reads only grid/block fields, so it would drop the
    # attributes.
    # CHECK: this target does not support launch attributes
    queue.execute[scale](
        Float32(1),
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=8),
        execution_config=AttributeConfig(),
    )


def main() raises:
    var driver = Driver.create("")
    var device = driver[].get_device[0]()
    var ctx = device[].get_context()
    var queue = ctx[].create_queue()
    launch(queue[])
