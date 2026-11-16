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
# Kernel stubs whose `execute` takes `ManagedTensorSlice` arguments, which the
# kernel verifier rejects in favor of tensor trait arguments. Kept apart from
# the valid testing kernels because the package as a whole fails verification.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    Input,
    InputTensor,
    InputVariadicTensors,
    Output,
    OutputTensor,
    Tensor,
)


@extensibility.register("mts_execute")
struct MtsExecute:
    @staticmethod
    def execute(z: OutputTensor, x: InputTensor, ctx: DeviceContext):
        pass


# A trait output does not hide a `ManagedTensorSlice` input after it.
@extensibility.register("mts_execute_mixed")
struct MtsExecuteMixed:
    @staticmethod
    def execute(z: Some[Output & Tensor], x: InputTensor, ctx: DeviceContext):
        pass


@extensibility.register("mts_execute_variadic")
struct MtsExecuteVariadic:
    @staticmethod
    def execute(
        z: Some[Output & Tensor],
        xs: InputVariadicTensors,
        ctx: DeviceContext,
    ):
        pass
