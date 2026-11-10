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
# Kernel stubs registered with platforms, `register(name, cond)`,
# for the import and resolution tests in
# GraphCompiler/test/mo-opt/MOGGDialect/Transforms/import-platforms.mlir.

from std.collections.string import StaticString
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    ACCELERATOR,
    CPU,
    CUDA,
    HIP,
    Input,
    Output,
    TileTensorable,
    api,
    arch,
    model,
)

comptime SM100 = ACCELERATOR & CUDA & arch("sm_100a")
comptime TCGEN05 = ACCELERATOR & CUDA & (
    arch("sm_100a") | arch("sm_101a") | arch("sm_103a")
)
comptime B200 = ACCELERATOR & CUDA & SM100 & model("NVIDIA B200")
comptime HOPPER_BLACKWELL = ACCELERATOR & CUDA & (
    arch("sm_90a") | arch("sm_100a")
)
comptime BLACKWELL = ACCELERATOR & CUDA & (arch("sm_100a") | arch("sm_103a"))
comptime TPU = ACCELERATOR & api("tpu")
comptime TPU_V6E = ACCELERATOR & TPU & arch("tpu-v6e")
comptime MY_ACCELERATORS = ACCELERATOR & (CUDA | HIP | TPU)


# `pred_matmul`: a generic accelerator kernel with narrower CUDA, HIP, and TPU
# kernels. On a B200, `TCGEN05` is a strict subset of `ACCELERATOR` and wins. On
# a TPU v6e, `TPU_V6E` is a strict subset of `TPU` and wins.
@extensibility.register("pred_matmul", CPU)
struct PredMatmulCpu:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_matmul", ACCELERATOR)
struct PredMatmulAccelerator:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_matmul", TCGEN05)
struct PredMatmulTcgen05:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_matmul", HIP)
struct PredMatmulHip:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_matmul", TPU)
struct PredMatmulTpu:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_matmul", TPU_V6E)
struct PredMatmulTpuV6e:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_b200", B200)
struct PredB200:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_my_accelerators", MY_ACCELERATORS)
struct PredMyAccelerators:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# `pred_mixed`: a registration for all devices and a platform registration.
# `TCGEN05` is narrower.
@extensibility.register("pred_mixed")
struct PredMixedAllDevices:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_mixed", TCGEN05)
struct PredMixedTcgen05:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# `pred_overlap`: two platforms that overlap on sm_100a, and neither is
# narrower. Selection on sm_100a is ambiguous.
@extensibility.register("pred_overlap", HOPPER_BLACKWELL)
struct PredOverlapHopperBlackwell:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_overlap", BLACKWELL)
struct PredOverlapBlackwell:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# `pred_specialized`: the same overlap, plus a kernel for exactly the overlap,
# which is narrower than both and wins on sm_100a.
@extensibility.register("pred_specialized", HOPPER_BLACKWELL)
struct PredSpecializedHopperBlackwell:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_specialized", BLACKWELL)
struct PredSpecializedBlackwell:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_specialized", SM100)
struct PredSpecializedSm100:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass
