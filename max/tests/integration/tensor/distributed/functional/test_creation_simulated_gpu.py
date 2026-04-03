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
"""Creation ops tests on simulated GPU mesh (single GPU, 4 virtual devices)."""

from _test_helpers import gpu_partial
from max.driver import Accelerator
from max.experimental.sharding import DeviceMesh

from functional._creation import CreationTests

gpu = Accelerator(0)


class TestCreation(CreationTests):
    MESH_1D = DeviceMesh(
        devices=(gpu, gpu, gpu, gpu),
        mesh_shape=(4,),
        axis_names=("tp",),
    )
    MESH_2D = DeviceMesh(
        devices=(gpu, gpu, gpu, gpu),
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
    )
    partial_fn = staticmethod(gpu_partial)
