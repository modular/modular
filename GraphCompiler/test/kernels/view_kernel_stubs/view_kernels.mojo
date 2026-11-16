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
# Whole-tensor kernels for the view ops, the way a backend whose kernels take
# whole tensors registers them (a concat, for example, as a DMA copy). MOToMAP
# calls such a kernel as an opaque kernel and lowers the op to its index loop
# only where no kernel matches, so the registrations are scoped to GPU and the
# same graph on the host still takes the loop. Kept apart from testing_kernels,
# which the GPU integration tests load: there these registrations would
# capture every broadcast, transpose and concat.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import Input, Output, Tensor, TensorTuple


@extensibility.register("mo.static.broadcast_to", extensibility.ACCELERATOR)
struct StaticBroadcastTo:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("mo.transpose", extensibility.ACCELERATOR)
struct Transpose:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        permutations: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("mo.concat", extensibility.ACCELERATOR)
struct Concat:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Tensor, ...],
        //,
        axis: Int,
    ](
        output: Some[Output & Tensor],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        pass
