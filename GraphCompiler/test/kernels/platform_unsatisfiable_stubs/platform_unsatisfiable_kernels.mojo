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
#
# A kernel stub whose platform holds for no device: the API of a device
# cannot be both "cuda" and "hip". The graph compiler must reject it.

from std.collections.string import StaticString
from max.gpu.host import DeviceContext

import extensibility
from extensibility import ACCELERATOR, CUDA, HIP, Input, Output, TileTensorable

comptime CUDA_AND_HIP = ACCELERATOR & CUDA & HIP


@extensibility.register("pred_unsatisfiable", CUDA_AND_HIP)
struct PredUnsatisfiable:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass
