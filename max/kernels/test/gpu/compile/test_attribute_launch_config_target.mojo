# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# A launch with an NVIDIA attribute configuration compiles only for a target
# that supports every capability the configuration uses.

# RUN: mkdir -p %t
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-cpu sapphirerapids \
# RUN:   --target-accelerator nvidia:sm_100a \
# RUN:   -D USE_CLUSTER=True \
# RUN:   -o %t/sm100.s %s
# RUN: %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-cpu sapphirerapids \
# RUN:   --target-accelerator nvidia:sm_80 \
# RUN:   -o %t/sm80.s %s
# RUN: not %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-cpu sapphirerapids \
# RUN:   --target-accelerator nvidia:sm_80 \
# RUN:   -D USE_CLUSTER=True \
# RUN:   -o %t/sm80_cluster.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-CLUSTER
# RUN: not %mojo-build --emit asm \
# RUN:   --target-triple x86_64-unknown-linux-gnu \
# RUN:   --target-cpu sapphirerapids \
# RUN:   --target-accelerator amdgpu:gfx942 \
# RUN:   -o %t/gfx942.s %s 2>&1 | FileCheck %s --check-prefix=CHECK-ATTRIBUTES

# CHECK-CLUSTER: {{this target does not support thread block cluster launch|cluster launch is not supported on this device}}
# CHECK-ATTRIBUTES: this target does not support launch attributes

from std.sys import get_defined_bool

from max.gpu.host import DeviceContext, Dim
from max.gpu.host.nvidia import (
    ClusterAttributeLaunchConfig,
    GridBlockAttributeLaunchConfig,
)
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes


def kernel():
    pass


def main() raises:
    with DeviceContext() as ctx:
        comptime if get_defined_bool["USE_CLUSTER", False]():
            ctx.enqueue_function[kernel](
                ClusterAttributeLaunchConfig(
                    grid_dim=2,
                    block_dim=32,
                    cluster_dim=Dim(2, 1, 1),
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
            )
        else:
            ctx.enqueue_function[kernel](
                GridBlockAttributeLaunchConfig(
                    grid_dim=1,
                    block_dim=32,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
            )
