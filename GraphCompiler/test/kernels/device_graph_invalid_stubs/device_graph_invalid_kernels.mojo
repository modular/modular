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
# stable-tensor arguments (`StableTensor` / `StableVariadicTensors`) where
# they are forbidden, and `build_graph` signatures that disagree with
# `execute` on which tensors are fused and in which form. Each violates
# exactly one rule; `mogg-verify-kernels` must reject every one. Kept apart
# from the valid testing kernels because the package as a whole fails
# verification.
#
from max.gpu.host import DeviceContext, DeviceGraphBuilder

import extensibility
from extensibility import (
    IOSpec,
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    StableInputTensor,
    StableInputVariadicTensors,
    StableTensor,
    _FusedComputeOutputTensor,
    _FusedInputTensor as FusedInputTensor,
    _FusedOutputTensor as FusedOutputTensor,
)


# A stable argument on `execute`: only `build_graph` receives slot addresses.
@extensibility.register("stable_on_execute")
struct StableOnExecute:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass


# A stable output on `build_graph`: outputs are written through live
# addresses.
@extensibility.register("stable_output_on_build_graph")
struct StableOutputOnBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: StableTensor[
            io_spec=IOSpec.Output, dtype=dtype, rank=rank, ...
        ],
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
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: FusedInputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: StableInputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# `execute` declares the output fused where `build_graph` does not: the
# epilogue functor built for `execute` has no plain parameter to land in.
@extensibility.register("fused_output_plain_build_graph")
struct FusedOutputPlainBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int, has_epilogue_fusion: Bool
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass


# Both entry points declare the output fused, but `execute` binds it through
# the compute entry point and `build_graph` through the store one, which are
# distinct `ManagedTensorSlice` types.
@extensibility.register("fused_compute_output_store_build_graph")
struct FusedComputeOutputStoreBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int, has_epilogue_fusion: Bool
    ](
        output: _FusedComputeOutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int, has_epilogue_fusion: Bool
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass
