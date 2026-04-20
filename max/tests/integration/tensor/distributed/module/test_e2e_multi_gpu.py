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
"""E2E tests on real GPUs -- inherits shared logic from _e2e (no IR tests)."""

from max.driver import Accelerator
from max.experimental.sharding import DeviceMesh

from module._e2e import E2ETests


class TestE2E(E2ETests):
    MESH_1D = DeviceMesh(
        devices=(
            Accelerator(0),
            Accelerator(1),
            Accelerator(2),
            Accelerator(3),
        ),
        mesh_shape=(4,),
        axis_names=("tp",),
    )
    MESH_2D = DeviceMesh(
        devices=(
            Accelerator(0),
            Accelerator(1),
            Accelerator(2),
            Accelerator(3),
        ),
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
    )
    MESH_2 = DeviceMesh(
        devices=(Accelerator(0), Accelerator(1)),
        mesh_shape=(2,),
        axis_names=("tp",),
    )
