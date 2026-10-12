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
"""DP dynamic-batch tests on real GPUs — inherits logic from _dp_dynamic."""

from max.driver import Accelerator
from max.experimental.sharding import DeviceMesh

from module._dp_dynamic import DPDynamicTests


class TestDPDynamic(DPDynamicTests):
    MESH_DP = DeviceMesh(
        devices=(Accelerator(0), Accelerator(1)),
        mesh_shape=(2,),
        axis_names=("dp",),
    )
