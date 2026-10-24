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
# Declarative kernel stubs (`execute_device` and `execute_configuration`) that
# each break one rule `mogg-verify-kernels` enforces. Kept apart from the valid
# testing kernels because the package as a whole fails verification. No launch
# is emitted for them, so `execute_configuration` returns a plain `Int`.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    Fused,
    Input,
    InputTensor,
    Output,
    OutputTensor,
    StoreAccess,
    Tensor,
    TileTensorable,
)


@extensibility.register("declarative_execute_configuration_only")
struct ExecuteConfigurationOnly:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0


@extensibility.register("declarative_execute_configuration_arity")
struct ExecuteConfigurationArity:
    @staticmethod
    def execute_configuration(a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Some[Input & TileTensorable]
    ):
        pass


@extensibility.register("declarative_execute_device_raises")
struct ExecuteDeviceRaises:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Some[Input & TileTensorable]
    ) raises:
        pass


@extensibility.register("declarative_execute_device_returns")
struct ExecuteDeviceReturns:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Some[Input & TileTensorable]
    ) -> Int:
        return 0


@extensibility.register("declarative_legacy_tensor")
struct LegacyTensorOperand:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(res: OutputTensor, a: InputTensor):
        pass


@extensibility.register("declarative_fused_operand")
struct FusedOperand:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & Fused & StoreAccess],
        a: Some[Input & TileTensorable],
    ):
        pass


@extensibility.register("declarative_operand_kinds")
struct OperandKindMismatch:
    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Scalar[DType.int32]
    ):
        pass


@extensibility.register("declarative_execute_and_execute_device")
struct ExecuteAndExecuteDevice:
    @staticmethod
    def execute[
        target: StaticString,
    ](res: OutputTensor, a: InputTensor, ctx: DeviceContext,) raises:
        pass

    @staticmethod
    def execute_configuration(res: Some[Tensor], a: Some[Tensor]) -> Int:
        return 0

    @staticmethod
    def execute_device(
        res: Some[Output & TileTensorable], a: Some[Input & TileTensorable]
    ):
        pass
