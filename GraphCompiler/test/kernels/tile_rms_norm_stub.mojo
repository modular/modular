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
# A tile kernel whose output is `_LegacyTileStoreAccess`, so the graph compiler
# fuses an elementwise consumer as a STORE epilogue (`OutputFusionTile`). The
# tile-fusion lit tests only inspect the IR and the emitted Mojo, so the body is
# never run; `tile_store_ops` in `max/tests` holds the kernel that runs.

from std.collections.string import StaticString

import extensibility
from extensibility import (
    Fused,
    Input,
    Output,
    TileTensorable,
    _LegacyTileStoreAccess,
)
from max.gpu.host import DeviceContext

comptime SM100 = (
    extensibility.ACCELERATOR
    & extensibility.CUDA
    & extensibility.arch("sm_100a")
)
"""Holds for CUDA devices with the `sm_100a` architecture."""


@extensibility.register("tile_rms_norm", SM100)
struct TileRMSNormStub:
    @staticmethod
    def execute(
        output: Some[Output & Fused & _LegacyTileStoreAccess],
        input: Some[Input & TileTensorable],
        gamma: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ):
        pass
