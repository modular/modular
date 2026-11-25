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

from max.gpu.host import DeviceContext


def scale(x: Float32, result: MutPointer[Float32, MutUntrackedOrigin]):
    result[] = 2 * x


def main() raises:
    var ctx = DeviceContext()
    # CHECK: the kernel takes 2 arguments but the launch passes 1
    ctx.enqueue_function[scale](Float32(1), grid_dim=1, block_dim=1)
