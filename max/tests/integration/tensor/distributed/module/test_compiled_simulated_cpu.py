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
"""Compiled tests on simulated CPU mesh -- inherits all logic from _compiled."""

from max.driver import CPU
from max.experimental.sharding import DeviceMesh

from module._compiled import CompiledTests


class TestCompiled(CompiledTests):
    MESH_2D = DeviceMesh(
        devices=(CPU(), CPU(), CPU(), CPU()),
        mesh_shape=(2, 2),
        axis_names=("dp", "tp"),
    )
    MESH_2 = DeviceMesh(
        devices=(CPU(), CPU()),
        mesh_shape=(2,),
        axis_names=("tp",),
    )
