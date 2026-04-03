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
"""Matmul tests on simulated CPU mesh — inherits all logic from _matmul."""

from _test_helpers import make_partial
from max.driver import CPU
from max.experimental.sharding import DeviceMesh

from functional._matmul import MatmulTests


class TestMatmul(MatmulTests):
    MESH_1D = DeviceMesh(
        devices=(CPU(), CPU(), CPU(), CPU()),
        mesh_shape=(4,),
        axis_names=("tp",),
    )
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
    partial_fn = staticmethod(make_partial)
