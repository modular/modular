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
"""Compiled tests on simulated GPU mesh (single GPU, virtual devices)."""

from max.driver import Accelerator
from max.experimental.sharding import DeviceMesh

from module._compiled import CompiledTests

gpu = Accelerator(0)


class TestCompiled(CompiledTests):
    MESH_2D = DeviceMesh(
        devices=(gpu, gpu, gpu, gpu),
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
    )
    MESH_2 = DeviceMesh(
        devices=(gpu, gpu),
        mesh_shape=(2,),
        axis_names=("tp",),
    )
