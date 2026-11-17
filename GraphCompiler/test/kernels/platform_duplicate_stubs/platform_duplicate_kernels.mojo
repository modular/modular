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
# Pairs of kernel stubs whose registrations are written differently but apply
# to the same devices. Neither kernel of a pair can ever be selected, so the
# graph compiler must report each pair as a duplicate.

from std.collections.string import StaticString
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    ACCELERATOR,
    CUDA,
    HIP,
    Input,
    Output,
    TileTensorable,
    arch,
)


# KGEN folds `CUDA | HIP` and `HIP | CUDA` to the same expression.
comptime CUDA_OR_HIP = ACCELERATOR & (CUDA | HIP)
comptime HIP_OR_CUDA = ACCELERATOR & (HIP | CUDA)


@extensibility.register("pred_dup_reordered", CUDA_OR_HIP)
struct PredDupCudaHip:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_dup_reordered", HIP_OR_CUDA)
struct PredDupHipCuda:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# KGEN does not distribute `&` over `|`, so these expressions differ, but they
# apply to the same devices.
comptime FACTORED = ACCELERATOR & CUDA & (arch("sm_100a") | arch("sm_103a"))
comptime DISTRIBUTED = (ACCELERATOR & CUDA & arch("sm_100a")) | (
    ACCELERATOR & CUDA & arch("sm_103a")
)


@extensibility.register("pred_dup_distributed", FACTORED)
struct PredDupFactored:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_dup_distributed", DISTRIBUTED)
struct PredDupDistributed:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# Two registrations without a platform both apply to all devices.
@extensibility.register("pred_dup_all_devices")
struct PredDupAllDevices:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_dup_all_devices")
struct PredDupAllDevicesAgain:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass
