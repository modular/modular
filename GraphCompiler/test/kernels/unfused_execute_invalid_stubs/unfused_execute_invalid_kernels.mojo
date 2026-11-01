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
# Kernel stubs whose second `execute` overload breaks one of the rules
# `mogg-import-packages` enforces on an `execute` overload pair. Each violates
# exactly one rule; import must reject every one. Kept apart from the valid
# testing kernels because the package as a whole fails import.
#
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    Fused,
    Input,
    LoadAccess,
    Output,
    StoreAccess,
    Tensor,
    TileTensorable,
    _LegacyComputeOutput,
)
from extensibility import InputTensor, OutputTensor
from extensibility import _FusedOutputTensor as FusedOutputTensor


# Not on the grandfather list.
@extensibility.register("unfused_not_allowed")
struct UnfusedNotAllowed:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Tensor], x: Some[Input & Tensor]) where not conforms_to(
        type_of(z), Fused
    ):
        pass


# A listed registration name, but on a device the entry does not name.
@extensibility.register(
    "trait_unfused_execute", type="gpu", api="cuda", arch="sm_100a"
)
struct UnfusedNotAllowedDevice:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Tensor], x: Some[Input & Tensor]) where not conforms_to(
        type_of(z), Fused
    ):
        pass


# Both overloads declare a fused output.
@extensibility.register("unfused_invalid_both_fused")
struct UnfusedInvalidBothFused:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & _LegacyComputeOutput], x: Some[Input & Tensor]):
        pass


# Three overloads.
@extensibility.register("unfused_invalid_three")
struct UnfusedInvalidThree:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Tensor], x: Some[Input & Tensor]) where not conforms_to(
        type_of(z), Fused
    ):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](
        z: Some[Output & TileTensorable], x: Some[Input & Tensor]
    ) where not conforms_to(type_of(z), Fused):
        pass


# The unfused overload's input differs.
@extensibility.register("unfused_invalid_input")
struct UnfusedInvalidInput:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](
        z: Some[Output & Tensor], x: Some[Input & Fused & LoadAccess]
    ) where not conforms_to(type_of(z), Fused):
        pass


# The unfused overload drops an output.
@extensibility.register("unfused_invalid_output_count")
struct UnfusedInvalidOutputCount:
    @staticmethod
    def execute[
        target: StaticString
    ](
        z: Some[Output & Fused & StoreAccess],
        w: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Tensor], x: Some[Input & Tensor]) where not conforms_to(
        type_of(z), Fused
    ):
        pass


# The unfused overload raises and the fused one does not.
@extensibility.register("unfused_invalid_raises")
struct UnfusedInvalidRaises:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString
    ](
        z: Some[Output & Tensor], x: Some[Input & Tensor]
    ) raises where not conforms_to(type_of(z), Fused):
        pass


# The unfused overload declares an extra parameter.
@extensibility.register("unfused_invalid_params")
struct UnfusedInvalidParams:
    @staticmethod
    def execute[
        target: StaticString
    ](z: Some[Output & Fused & StoreAccess], x: Some[Input & Tensor]):
        pass

    @staticmethod
    def execute[
        target: StaticString, extra: Int
    ](z: Some[Output & Tensor], x: Some[Input & Tensor]) where not conforms_to(
        type_of(z), Fused
    ):
        pass


# Legacy `ManagedTensorSlice` argument types.
@extensibility.register("unfused_invalid_legacy")
struct UnfusedInvalidLegacy:
    @staticmethod
    def execute[target: StaticString](z: FusedOutputTensor, x: InputTensor):
        pass

    @staticmethod
    def execute[target: StaticString](z: OutputTensor, x: InputTensor):
        pass
