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
# Whole-tensor kernels for the builtin view ops, the way a backend whose
# kernels take whole tensors registers them. MOToMAP calls a registered view
# kernel as an opaque kernel and lowers the op to its index loop only where no
# kernel matches the device, so the registrations are scoped to GPU and the
# same graph on the host still takes the loop. Kept apart from
# testing_kernels, which the GPU integration tests load: there these
# registrations would capture every broadcast and transpose.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import InputTensor, OutputTensor


@extensibility.register("mo.static.broadcast_to", type="gpu")
struct StaticBroadcastTo:
    @staticmethod
    def execute(
        output: OutputTensor,
        input: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("mo.transpose", type="gpu")
struct Transpose:
    @staticmethod
    def execute(
        output: OutputTensor,
        input: InputTensor,
        permutations: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass
