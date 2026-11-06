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
# A tile kernel whose output is `_FusedOutputTileTensor`, so the graph compiler
# fuses an elementwise consumer as a STORE epilogue (`OutputFusionTile`). The
# tile-fusion lit tests only inspect the IR and the emitted Mojo, so the body is
# never run; `tile_store_ops` in `max/tests` holds the kernel that runs.

from std.collections.string import StaticString

import extensibility
from extensibility import InputTensor, _FusedOutputTileTensor
from max.gpu.host import DeviceContext


@extensibility.register("tile_rms_norm", type="gpu", api="cuda", arch="sm_100a")
struct TileRMSNormStub:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: _FusedOutputTileTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ):
        pass
