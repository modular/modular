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
# Kernel stubs that break a device-graph rule the kernel verifier enforces:
# legacy tensor structs on `build_graph`, stable arguments where they are
# forbidden, and `build_graph` signatures that disagree with `execute` on which
# tensors are fused or stable, and in which form. Each violates exactly one rule; `mogg-verify-kernels` must reject every
# one. Kept apart from the valid testing kernels because the package as a
# whole fails verification.
#
from max.gpu.host import DeviceContext, DeviceGraphBuilder

import extensibility
from extensibility import (
    Fused,
    Input,
    InputTensor,
    LoadAccess,
    Output,
    Stable,
    StoreAccess,
    Tensor,
    _LegacyComputeOutput,
)


# A `ManagedTensorSlice` on `build_graph`: the graph compiler routes only
# tensor trait arguments through stable address slots.
@extensibility.register("mts_on_build_graph")
struct MtsOnBuildGraph:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: Some[Output & Tensor],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# A stable argument over a fused `execute` input: the slot publishes the
# tensor's address to peers, and a fused input is a functor the kernel
# evaluates rather than a tensor in memory, so there is no address to publish.
@extensibility.register("stable_over_fused_execute")
struct StableOverFusedExecute:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Fused & LoadAccess],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor & Stable],
        builder: DeviceGraphBuilder,
    ):
        pass


# A trait `Stable` argument on `execute`: only `build_graph` operands are
# routed through a slot, so `execute` would read a data pointer as a slot
# address. Rejected whether the kernel has no `build_graph` or one taking the
# argument `Stable` or plain.
@extensibility.register("trait_stable_on_execute_only")
struct TraitStableOnExecuteOnly:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor & Stable],
        ctx: DeviceContext,
    ):
        pass


@extensibility.register("trait_stable_on_execute")
struct TraitStableOnExecute:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor & Stable],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor & Stable],
        builder: DeviceGraphBuilder,
    ):
        pass


@extensibility.register("trait_stable_on_execute_plain_build_graph")
struct TraitStableOnExecutePlainBuildGraph:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor & Stable],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        builder: DeviceGraphBuilder,
    ):
        pass


# `execute` declares the output fused where `build_graph` does not: the
# epilogue functor built for `execute` has no plain parameter to land in.
@extensibility.register("fused_output_plain_build_graph")
struct FusedOutputPlainBuildGraph:
    @staticmethod
    def execute[
        has_epilogue_fusion: Bool
    ](
        output: Some[Output & Fused & StoreAccess],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        builder: DeviceGraphBuilder,
    ):
        pass


# Both entry points declare the output fused, but `execute` binds it through
# the compute entry point and `build_graph` through the store one, which carry
# different access contracts.
@extensibility.register("fused_compute_output_store_build_graph")
struct FusedComputeOutputStoreBuildGraph:
    @staticmethod
    def execute[
        has_epilogue_fusion: Bool
    ](
        output: Some[Output & Fused & _LegacyComputeOutput],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        has_epilogue_fusion: Bool
    ](
        output: Some[Output & Fused & StoreAccess],
        x: Some[Input & Tensor],
        builder: DeviceGraphBuilder,
    ):
        pass
