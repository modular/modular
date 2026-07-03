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
"""Collective ops tests on real 4 GPUs — inherits all logic from _collectives."""

from _test_helpers import make_partial
from max.driver import Accelerator
from max.experimental.sharding import DeviceMesh

from functional._collectives import CollectivesTests


class TestCollectives(CollectivesTests):
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
    partial_fn = staticmethod(make_partial)
