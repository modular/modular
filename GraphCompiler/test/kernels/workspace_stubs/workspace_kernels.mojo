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
# Kernels that request a workspace buffer: a `Workspace` argument on `execute` or
# `execute_device`, sized by `workspace_size`. They only exercise import and
# `mogg-verify-kernels`, so their bodies are empty.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    Input,
    MutableInput,
    Output,
    Tensor,
    TensorTuple,
    TileTensorable,
    Workspace,
)


@extensibility.register("workspace_execute")
struct WorkspaceExecute:
    @staticmethod
    def workspace_size(a: Some[Tensor], b: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        b: Some[MutableInput & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass


# `execute_configuration` takes the kernel's operands, and no workspace buffer.
@extensibility.register("workspace_execute_device")
struct WorkspaceExecuteDevice:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
    ):
        pass


# `workspace_size` takes a `TensorTuple` where the entry point does, and no
# argument for a scalar input.
@extensibility.register("workspace_variadic")
struct WorkspaceVariadic:
    @staticmethod
    def workspace_size[
        Xs: TypeList[Trait=Tensor, ...],
        //,
    ](xs: TensorTuple[*Xs], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute[
        Xs: TypeList[Trait=Input & TileTensorable, ...],
        //,
    ](
        res: Some[Output & TileTensorable],
        xs: TensorTuple[*Xs],
        a: Some[Input & TileTensorable],
        scale: Float32,
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass
