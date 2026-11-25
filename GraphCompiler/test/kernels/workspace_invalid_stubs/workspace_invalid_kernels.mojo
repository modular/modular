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
# Kernels that each break one rule `mogg-verify-kernels` enforces on workspace
# buffers. Kept apart from the valid stubs because the package as a whole fails
# verification.
#
from max.gpu.host import DeviceContext, DeviceGraphBuilder

import extensibility
from extensibility import (
    Input,
    Output,
    Tensor,
    TileTensorable,
    Workspace,
)


@extensibility.register("workspace_without_workspace_size")
struct WithoutWorkspaceSize:
    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("workspace_size_without_workspace")
struct WorkspaceSizeWithoutWorkspace:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("workspace_twice")
struct WorkspaceTwice:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        more: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("workspace_before_operand")
struct WorkspaceBeforeOperand:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        workspace: Some[Workspace],
        a: Some[Input & TileTensorable],
        ctx: DeviceContext,
    ) raises:
        pass


@extensibility.register("workspace_after_context")
struct WorkspaceAfterContext:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        ctx: DeviceContext,
        workspace: Some[Workspace],
    ) raises:
        pass


@extensibility.register("workspace_with_build_graph")
struct WorkspaceWithBuildGraph:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass

    @staticmethod
    def build_graph(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        builder: DeviceGraphBuilder,
    ) raises:
        pass


@extensibility.register("workspace_size_returns_float")
struct WorkspaceSizeReturnsFloat:
    @staticmethod
    def workspace_size(a: Some[Tensor]) -> Float32:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass


# `workspace_size` takes the inputs, and not the output.
@extensibility.register("workspace_size_arity")
struct WorkspaceSizeArity:
    @staticmethod
    def workspace_size(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass


# `TileTensorable` gives `workspace_size` access to the input's data.
@extensibility.register("workspace_size_reads_data")
struct WorkspaceSizeReadsData:
    @staticmethod
    def workspace_size(a: Some[TileTensorable]) -> Int:
        return 0

    @staticmethod
    def execute(
        res: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        workspace: Some[Workspace],
        ctx: DeviceContext,
    ) raises:
        pass
